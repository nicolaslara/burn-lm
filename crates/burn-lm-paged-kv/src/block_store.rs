use burn::tensor::{Device, IndexingUpdateOp, Int, Tensor};

use crate::cache::LanePlan;

#[derive(Debug, Clone)]
/// A fixed-size pool of KV blocks, shaped `[num_blocks, block_size, num_heads, head_dim]` —
/// token-major, so a logical position is one row of the leading two dims and a whole round's
/// writes land in ONE indexed scatter, regardless of how many lanes wrote or how many block
/// boundaries their tokens cross. This type is purely physical: it neither knows which lane owns
/// which block nor tracks any lengths — the caller hands every write its destination indices and
/// every read the block ids to gather, so the same store serves any lane-to-block assignment.
///
/// Block 0 is a zeroed sentinel no lane ever owns (the `BlockPool` never allocates it). Ragged
/// gathers pad short lanes with it, so padding can never read a live block — even a masking bug
/// then exposes zeros, not another sequence's KV.
pub struct BlockStore {
    /// `[num_blocks, block_size, num_heads, head_dim]`; block 0 is the sentinel.
    pool: Tensor<4>,
}

impl BlockStore {
    /// Creates an empty store of `num_blocks` blocks spanning `block_size` positions each. Only
    /// the sentinel (block 0) is initialized, to zeros; every other block holds garbage until
    /// written, exactly like the old slab.
    pub(crate) fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Self {
        let pool = Tensor::empty([num_blocks, block_size, num_heads, head_dim], device);
        let zeros = Tensor::zeros([1, block_size, num_heads, head_dim], device);
        let pool = pool.slice_assign([0..1, 0..block_size, 0..num_heads, 0..head_dim], zeros);
        Self { pool }
    }

    /// Tokens per block. Only the tests need to ask: everything else is handed its indices.
    #[cfg(test)]
    fn block_size(&self) -> usize {
        self.pool.shape()[1]
    }

    /// Write one round's new tokens in a single indexed scatter. `rows` is the round's fresh K or
    /// V, `[n, num_heads, seq_len, head_dim]` (one lane per row, as the projection produces it);
    /// `write_idx` is `[n·seq_len, 2]` of `(block, offset)` destinations in lane-major, position
    /// order — built once per round by `prepare_lanes` and shared by every layer's K and V. Tokens
    /// crossing block boundaries are nothing special: a boundary is just a different index pair.
    ///
    /// The indices are unique by construction (each destination is written exactly once per
    /// round), which is what makes `Assign` well-defined. One kernel launch per call, whatever the
    /// width — this is the generic-op stand-in for a dedicated cache-write kernel (vLLM's
    /// `reshape_and_cache`), and it already takes that kernel's exact inputs.
    pub(crate) fn write(&mut self, write_idx: &Tensor<2, Int>, rows: Tensor<4>) {
        let [n, heads, seq_len, head_dim] = rows.dims();
        // [n, heads, seq, d] -> [n·seq, heads, d]: token-major to match the pool layout.
        let rows = rows.swap_dims(1, 2).reshape([n * seq_len, heads, head_dim]);
        let idx = write_idx.clone();
        self.pool
            .inplace(|pool| pool.scatter_nd(idx, rows, IndexingUpdateOp::Assign));
    }

    /// The raw pool tensor, `[num_blocks, block_size, num_heads, head_dim]`.
    ///
    /// For the paged-attention kernel, which addresses the blocks in place instead of gathering
    /// them. The returned value is a handle (a refcount bump), not a copy — and that is exactly why
    /// it must not be held: `write` mutates through `inplace`/`scatter_nd`, which only skips a full
    /// copy while the pool handle is uniquely owned, so a clone kept alive across a later round's
    /// writes turns every KV write into a copy-on-write of the whole pool. Take it, use it, drop
    /// it, in that order, inside one call. Same contract as the clone in
    /// [`gather`](Self::gather).
    pub(crate) fn pool(&self) -> RaggedKv {
        RaggedKv(self.pool.clone())
    }

    /// Read `l_max` positions for each of `n` lanes as one `[n, num_heads, l_max, head_dim]`
    /// tensor, from a caller-built gather index: `idx` holds `n · blocks_per_lane` block ids, each
    /// lane's covering blocks in position order, short lanes padded with the sentinel. The index is
    /// a pure function of the round's tables, so the caller (`prepare_lanes`) builds and uploads it
    /// once per round and every layer's K and V reuse the same handle.
    ///
    /// The blocks are selected in one indexed gather, stitched back into a contiguous sequence, and
    /// trimmed to `l_max`. Whole blocks are selected before the trim (the granularity cost of
    /// paging, at most `block_size - 1` wasted columns per lane), and this leans on the backend
    /// fusing the chain; the decode latency gate measures whether it does.
    ///
    /// The lanes sit at independent positions, so shorter lanes come back with a stale tail past
    /// their own length. The caller MUST mask that tail with the per-lane padding mask; this store
    /// does not zero it.
    pub(crate) fn gather(
        &self,
        idx: Tensor<1, Int>,
        blocks_per_lane: usize,
        l_max: usize,
    ) -> RaggedKv {
        let [_, bs, heads, head_dim] = self.pool.dims();
        let nb = blocks_per_lane;
        let n = idx.dims()[0] / nb;
        // Select every lane's covering blocks, stitch each lane's blocks into one contiguous
        // sequence axis, then put heads ahead of positions for attention:
        // [n·nb, bs, h, d] -> [n, nb·bs, h, d] -> [n, h, nb·bs, d] -> trim to l_max.
        //
        // The clone is a handle (refcount bump), not a copy of the pool — and it must stay AFTER
        // the writes: `write` mutates through `inplace`/`scatter_nd`, which only skips a full copy
        // while the pool handle is uniquely owned. A pool clone held across the writes would turn
        // every layer's KV write into a copy-on-write of the whole pool.
        RaggedKv(
            self.pool
                .clone()
                .select(0, idx)
                .reshape([n, nb * bs, heads, head_dim])
                .swap_dims(1, 2)
                .slice([0..n, 0..heads, 0..l_max, 0..head_dim]),
        )
    }
}

/// KV read out of the block pool, with the positions nobody ever wrote still in it.
///
/// [`BlockStore::new`] allocates the pool with `Tensor::empty` and zeroes only the sentinel block.
/// From then on the only elements anything writes are a lane's own positions `[0, len)`. Every
/// other element — the dead tail of a lane's last block, and the whole of any block the pool has
/// not handed out yet — is whatever the allocator had lying around. That is not "stale KV from an
/// earlier sequence"; on the first pass through a fresh pool it is arbitrary bit patterns, and an
/// arbitrary 32-bit pattern is a NaN or an infinity about one time in 256.
///
/// **Masking is not a defence against those bytes.** A mask correctly gives a dead column zero
/// attention weight, and the value aggregation then computes `0 · V_dead` — which is NaN when
/// `V_dead` is NaN, because that is what IEEE arithmetic says. One NaN in an unwritten tail turns
/// the whole output row NaN, and from there the whole forward.
///
/// So a read does not hand back a plain tensor; it hands back this, and there are exactly two ways
/// out, mirroring `MaybeUninit`/`assume_init`:
///
/// - [`neutralized`](Self::neutralized) — write finite zeros over every dead column, for a
///   consumer that touches whole blocks and masks afterwards (the tensor-op reference).
/// - [`assume_length_gated`](Self::assume_length_gated) — the unchecked exit, for a consumer that
///   provably never looks past a lane's length (the decode kernel).
///
/// Both layouts the pool is read in are covered: the gathered per-lane scratch from
/// [`BlockStore::gather`], `[n, kv_heads, l_max, head_dim]`, and the raw pool from
/// [`BlockStore::pool`], `[num_blocks, block_size, kv_heads, head_dim]`. What the type tracks is
/// where the bytes came from, not how they are shaped — and they came from memory nobody
/// initialized. Only the gathered layout can be neutralized, because only it has a lane per row
/// for the plan's mask to line up against.
#[must_use = "this KV still holds never-written positions: `neutralized` zeroes them, \
              `assume_length_gated` asserts this consumer stops at each lane's length"]
pub struct RaggedKv(Tensor<4>);

impl RaggedKv {
    /// Write finite zeros over every position past each lane's own length, and hand back the
    /// tensor.
    ///
    /// This is the exit for a consumer that reads whole blocks and relies on a mask — which cannot
    /// work on its own, for the `0 · NaN` reason on the type. Zeros are the right filler: they are
    /// what the sentinel block already holds, they keep the masked softmax's arithmetic finite,
    /// and the mask still removes the columns afterwards, so nothing about the *answer* depends on
    /// the value chosen.
    ///
    /// Callers neutralize both K and V even though only V can carry the NaN into the answer — a
    /// dead K feeds a score the mask overwrites. Zeroing K as well costs one more pass over the
    /// same-sized tensor and buys a rule that fits in one sentence: nothing past a length is ever
    /// handed to the reference. A rule with an exception in it is the kind that gets misapplied.
    ///
    /// Costs a quarter of what it would after grouped-query expansion, and callers must keep it
    /// that way: this runs on the `[n, kv_heads, l_max, head_dim]` gather, before `repeat_kv`
    /// turns `kv_heads` into `kv_heads · n_rep`.
    pub fn neutralized(self, plan: &LanePlan) -> Tensor<4> {
        let [n, heads, l_max, head_dim] = self.0.dims();
        let [mask_n, _, seq_q, mask_l_max] = plan.mask.dims();
        debug_assert_eq!(
            [mask_n, mask_l_max],
            [n, l_max],
            "neutralized got a plan that does not describe this gather"
        );
        // The last query row is the one whose live columns are the lane's whole history: row `r`
        // may attend to `0..=starts[j] + r`, so row `seq_q - 1` masks exactly the columns at or
        // past `starts[j] + seq_q`, which is the lane's length after this round. Earlier rows also
        // hide the lane's own future, and those columns are live memory that must survive.
        let dead = plan
            .mask
            .clone()
            .slice([0..n, 0..1, seq_q - 1..seq_q, 0..l_max])
            .reshape([n, 1, l_max, 1])
            .expand([n, heads, l_max, head_dim]);
        self.0.mask_fill(dead, 0.0)
    }

    /// Take the tensor as it is, never-written positions included.
    ///
    /// Named after `MaybeUninit::assume_init`, and the assertion has the same shape. The caller is
    /// **not** claiming that anybody zeroed this memory — nobody did. It is claiming that it never
    /// reads the parts that were never written: for a lane of length `len` over blocks of
    /// `block_size`, that it walks exactly `min(block_size, len - b·block_size)` positions of
    /// block `b` and stops there, and that it never touches a block outside the lane's own table.
    ///
    /// Break that and there is no mask underneath to catch it. The bytes past a length are
    /// arbitrary, one NaN among them is enough to make an output row NaN, and on a pool that has
    /// churned they are a live sequence's keys and values instead.
    pub fn assume_length_gated(self) -> Tensor<4> {
        self.0
    }
}

/// Build the `(block, offset)` destination pairs for one round's writes: for each lane in order,
/// its `seq_len` new positions starting at `starts[j]`, translated through its block table. Plain
/// index arithmetic — a token past a block edge simply lands in the next table entry, which is how
/// chunked prefill's boundary crossings cost nothing special.
pub(crate) fn write_indices(
    tables: &[Vec<u32>],
    starts: &[usize],
    seq_len: usize,
    block_size: usize,
) -> Vec<i32> {
    let mut ids = Vec::with_capacity(tables.len() * seq_len * 2);
    for (table, &start) in tables.iter().zip(starts) {
        for p in start..start + seq_len {
            ids.push(table[p / block_size] as i32);
            ids.push((p % block_size) as i32);
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_pool::SENTINEL_BLOCK;
    use burn::tensor::TensorData;

    /// A `[1, 1, len, 1]` rows tensor holding `base + position` at each position, so any
    /// misplacement changes some element.
    fn vals(base: usize, range: std::ops::Range<usize>) -> Tensor<4> {
        let data: Vec<f32> = range.map(|p| (base + p) as f32).collect();
        let len = data.len();
        Tensor::<4>::from_data(TensorData::new(data, [1, 1, len, 1]), &Default::default())
    }

    /// The expected `[1, 1, n, 1]` read-back for `base + position` over `0..n`.
    fn expect(base: usize, n: usize) -> TensorData {
        TensorData::new(
            (0..n).map(|p| (base + p) as f32).collect::<Vec<f32>>(),
            [1, 1, n, 1],
        )
    }

    /// `write` through host-built indices, as `prepare_lanes` does per round.
    fn write(store: &mut BlockStore, tables: &[Vec<u32>], starts: &[usize], rows: Tensor<4>) {
        let seq_len = rows.dims()[2];
        let ids = write_indices(tables, starts, seq_len, store.block_size());
        let n = ids.len() / 2;
        let idx = Tensor::<2, Int>::from_data(TensorData::new(ids, [n, 2]), &Default::default());
        store.write(&idx, rows);
    }

    /// Read back through [`BlockStore::gather`].
    ///
    /// These tests are the length-gated consumer themselves: each one asserts only over the
    /// positions it wrote, and the one that reads a short lane's stale tail says so and does not
    /// assert on it. That is exactly the invariant [`RaggedKv::assume_length_gated`] names.
    fn gathered(
        store: &BlockStore,
        idx: Tensor<1, Int>,
        blocks_per_lane: usize,
        l_max: usize,
    ) -> Tensor<4> {
        store
            .gather(idx, blocks_per_lane, l_max)
            .assume_length_gated()
    }

    /// The gather index `prepare_lanes` would build: each table's blocks in order, sentinel-padded
    /// to `nb` entries per lane.
    fn idx_for(tables: &[Vec<u32>], nb: usize) -> Tensor<1, Int> {
        let ids: Vec<i32> = tables
            .iter()
            .flat_map(|t| (0..nb).map(|i| *t.get(i).unwrap_or(&SENTINEL_BLOCK) as i32))
            .collect();
        let n = ids.len();
        Tensor::from_data(TensorData::new(ids, [n]), &Default::default())
    }

    /// Lanes at divergent positions: ragged writes land at each lane's own caller-supplied offset,
    /// the read-back covers the longest active lane, and recycling a lane's block (offset 0)
    /// overwrites it without touching its siblings. Block ids are deliberately not `lane + 1` — the
    /// store must follow the indices it is handed, nothing else.
    #[test]
    fn test_writes_land_at_ragged_positions_and_blocks_recycle() {
        let device = crate::test_device::test_device();
        // [num_blocks=4 (sentinel + 3), block_size=8, heads=1, head_dim=2]
        let mut store = BlockStore::new(4, 8, 1, 2, &device);
        let t0 = vec![3u32]; // lane 0 -> block 3
        let t2 = vec![1u32]; // lane 2 -> block 1

        write(
            &mut store,
            std::slice::from_ref(&t0),
            &[0],
            Tensor::full([1, 1, 3, 2], 1.0, &device),
        );
        write(
            &mut store,
            std::slice::from_ref(&t2),
            &[0],
            Tensor::full([1, 1, 1, 2], 3.0, &device),
        );

        // Fused decode write: one new position per active lane at each lane's own offset (3 and 1).
        let step = Tensor::<4>::from_data([[[[10.0, 10.0]]], [[[30.0, 30.0]]]], &device);
        let tables = [t0.clone(), t2.clone()];
        write(&mut store, &tables, &[3, 1], step);
        let out = gathered(&store, idx_for(&tables, 1), 1, 4);
        assert_eq!(out.dims(), [2, 1, 4, 2]);
        out.clone()
            .slice([0..1, 0..1, 0..4, 0..2])
            .to_data()
            .assert_eq(
                &TensorData::from([[[[1.0f32, 1.0], [1.0, 1.0], [1.0, 1.0], [10.0, 10.0]]]]),
                false,
            );
        // Lane 2's columns 2..4 are stale tail the mask must cover — not asserted.
        out.slice([1..2, 0..1, 0..2, 0..2])
            .to_data()
            .assert_eq(&TensorData::from([[[[3.0f32, 3.0], [30.0, 30.0]]]]), false);

        // Lane 0's block is recycled from position 0, overwriting its old contents.
        write(
            &mut store,
            std::slice::from_ref(&t0),
            &[0],
            Tensor::full([1, 1, 2, 2], 7.0, &device),
        );
        let out = gathered(&store, idx_for(std::slice::from_ref(&t0), 1), 1, 2);
        out.to_data()
            .assert_eq(&TensorData::from([[[[7.0f32, 7.0], [7.0, 7.0]]]]), false);
    }

    /// KV-contents equivalence with the slab semantics this store replaced: a scripted mix of
    /// chunked prefills (writes at `position > 0`), interleaved lanes, and a fused decode write is
    /// read back and compared element-for-element against the tensor the slab rules dictate. The
    /// batched-equivalence suite checks logits; this checks the stored bytes directly, so a
    /// write-address or gather-order bug is caught before attention numerics can hide it.
    #[test]
    fn scripted_writes_reproduce_slab_contents_exactly() {
        let device = crate::test_device::test_device();
        let mut store = BlockStore::new(3, 6, 1, 1, &device);
        let t0 = vec![2u32];
        let t1 = vec![1u32];

        write(&mut store, std::slice::from_ref(&t0), &[0], vals(100, 0..2));
        write(&mut store, std::slice::from_ref(&t0), &[2], vals(100, 2..4));
        write(&mut store, std::slice::from_ref(&t1), &[0], vals(200, 0..3));

        let step = Tensor::<4>::from_data(
            TensorData::new(vec![104.0f32, 203.0], [2, 1, 1, 1]),
            &device,
        );
        let tables = [t0, t1];
        write(&mut store, &tables, &[4, 3], step);
        let out = gathered(&store, idx_for(&tables, 1), 1, 5);

        assert_eq!(out.dims(), [2, 1, 5, 1]);
        out.clone()
            .slice([0..1, 0..1, 0..5, 0..1])
            .to_data()
            .assert_eq(&expect(100, 5), false);
        out.slice([1..2, 0..1, 0..4, 0..1])
            .to_data()
            .assert_eq(&expect(200, 4), false);
    }

    /// A prefill chunk spanning several small blocks lands intact through ONE scatter: start offset
    /// 2 into a half-full tail block, 9 more tokens across blocks of 4, then decode tokens crossing
    /// into a fresh block. Boundary crossings are just different index pairs — there is no split
    /// logic left to test, only the arithmetic.
    #[test]
    fn writes_cross_block_boundaries_and_read_back_contiguously() {
        let device = crate::test_device::test_device();
        let mut store = BlockStore::new(5, 4, 1, 1, &device);
        // Deliberately unordered, non-contiguous ids: position i·4.. lives in table[i].
        let table = vec![3u32, 1, 4];

        write(
            &mut store,
            std::slice::from_ref(&table),
            &[0],
            vals(500, 0..2),
        );
        write(
            &mut store,
            std::slice::from_ref(&table),
            &[2],
            vals(500, 2..11),
        );
        let out = gathered(&store, idx_for(std::slice::from_ref(&table), 3), 3, 11);
        assert_eq!(out.dims(), [1, 1, 11, 1]);
        out.to_data().assert_eq(&expect(500, 11), false);

        write(
            &mut store,
            std::slice::from_ref(&table),
            &[11],
            vals(500, 11..12),
        );
        let grown = vec![3u32, 1, 4, 2];
        write(
            &mut store,
            std::slice::from_ref(&grown),
            &[12],
            vals(500, 12..13),
        );
        let out = gathered(&store, idx_for(std::slice::from_ref(&grown), 4), 4, 13);
        out.to_data().assert_eq(&expect(500, 13), false);
    }

    /// Ragged multi-lane gather with small blocks: the long lane sets `l_max`, the short lane's
    /// missing blocks come back as the zeroed sentinel — provably zeros, not another lane's data.
    #[test]
    fn short_lanes_pad_with_the_sentinel_never_a_live_block() {
        let device = crate::test_device::test_device();
        let mut store = BlockStore::new(5, 2, 1, 1, &device);
        let long = vec![1u32, 2]; // positions 0..4
        let short = vec![3u32]; // positions 0..2

        write(
            &mut store,
            std::slice::from_ref(&long),
            &[0],
            vals(700, 0..4),
        );
        write(
            &mut store,
            std::slice::from_ref(&short),
            &[0],
            vals(900, 0..1),
        );

        let long_grown = vec![1u32, 2, 4];
        let step = Tensor::<4>::from_data(
            TensorData::new(vec![704.0f32, 901.0], [2, 1, 1, 1]),
            &device,
        );
        let tables = [long_grown, short];
        write(&mut store, &tables, &[4, 1], step);
        let out = gathered(&store, idx_for(&tables, 3), 3, 5);
        assert_eq!(out.dims(), [2, 1, 5, 1]);
        out.clone()
            .slice([0..1, 0..1, 0..5, 0..1])
            .to_data()
            .assert_eq(&expect(700, 5), false);
        out.clone()
            .slice([1..2, 0..1, 0..2, 0..1])
            .to_data()
            .assert_eq(&expect(900, 2), false);
        out.slice([1..2, 0..1, 2..5, 0..1])
            .to_data()
            .assert_eq(&TensorData::new(vec![0.0f32; 3], [1, 1, 3, 1]), false);
    }

    /// The sentinel block is zeroed at construction and no write may touch it.
    #[test]
    fn sentinel_block_stays_zeroed() {
        let device = crate::test_device::test_device();
        let mut store = BlockStore::new(3, 4, 1, 1, &device);
        write(
            &mut store,
            &[vec![1u32], vec![2u32]],
            &[0, 0],
            Tensor::<4>::full([2, 1, 4, 1], 9.0, &device),
        );
        let sentinel = store.pool.clone().slice([0..1, 0..4, 0..1, 0..1]);
        sentinel
            .to_data()
            .assert_eq(&TensorData::new(vec![0.0f32; 4], [1, 4, 1, 1]), false);
    }
}
