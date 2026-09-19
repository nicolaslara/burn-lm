//! The plain-Rust twin of the kernel's recurrence.
//!
//! This is not a second implementation of attention — it is the *same* recurrence written where it
//! can be read: walk the lane's block table, stop at the lane's length, and update `(m, l, acc)` in
//! f32 with the identical rescale, from the identical finite sentinel. The only thing that differs
//! from the kernel is the summation order inside one `head_dim`-term dot product (the kernel splits
//! it across a plane and folds with `plane_sum`; this sums left to right), which is bounded at
//! roughly 1e-7 relative in f32.
//!
//! That is what makes it the tight oracle. A wrong rescale — the one bug an online softmax invites —
//! is O(1) wrong, not 1e-7 wrong, so a 1e-6 comparison against this catches it immediately. The
//! tensor-op reference cannot do that job as well: it runs its softmax in the storage dtype, so in
//! f16 it is the *less* accurate of the two sides and its tolerance has to be loose enough to hide
//! exactly the class of error this catches.

/// The geometry one decode round attends over. Named rather than passed as eleven positional
/// arguments, because every one of them is a stride multiplier and swapping two silently produces
/// plausible numbers.
#[derive(Debug, Clone, Copy)]
pub struct DecodeShape {
    /// Active lanes.
    pub n: usize,
    /// Key/value heads.
    pub num_kv_heads: usize,
    /// Query heads per KV head (`num_heads / num_kv_heads`).
    pub n_rep: usize,
    /// Elements per head.
    pub head_dim: usize,
    /// Token positions per block.
    pub block_size: usize,
    /// Block-table entries per lane.
    pub blocks_per_lane: usize,
    /// Softmax scale (`1/sqrt(head_dim)` for the default attention options).
    pub scale: f32,
}

/// The same finite sentinel the kernel uses. Not `-inf`: `exp(sentinel - s)` is exactly zero for
/// any real `s`, while `(-inf) - (-inf)` is NaN, and WGSL rejects an infinity literal outright.
const NEG_SENTINEL: f32 = -3.402_823_5e38;

/// Attend one query position per lane, walking the block table exactly as the kernel does.
///
/// - `q` is `[n, num_heads, head_dim]` (the singleton query axis dropped),
/// - `k_pool` / `v_pool` are `[num_blocks, block_size, num_kv_heads, head_dim]`,
/// - `block_table` is `[n · blocks_per_lane]`,
/// - `lengths` is `[n]`, and entries past `lengths[j]` in the table or in a live block's tail are
///   never read — which is the property the poison test exists to prove.
///
/// Returns `[n, num_heads, head_dim]`.
pub fn decode_reference_online(
    q: &[f32],
    k_pool: &[f32],
    v_pool: &[f32],
    block_table: &[i32],
    lengths: &[i32],
    shape: DecodeShape,
) -> Vec<f32> {
    let DecodeShape {
        n,
        num_kv_heads,
        n_rep,
        head_dim,
        block_size,
        blocks_per_lane,
        scale,
    } = shape;
    let num_heads = num_kv_heads * n_rep;
    assert_eq!(q.len(), n * num_heads * head_dim, "q shape");
    assert_eq!(lengths.len(), n, "lengths shape");
    assert_eq!(block_table.len(), n * blocks_per_lane, "block table shape");
    assert_eq!(k_pool.len(), v_pool.len(), "pools must match");

    let mut out = vec![0.0f32; n * num_heads * head_dim];

    for lane in 0..n {
        let len = lengths[lane] as usize;
        assert!(len >= 1, "a decode lane always holds at least one position");
        for kv_head in 0..num_kv_heads {
            // One plane's worth of work: `n_rep` output rows sharing every K and V read.
            let mut m = vec![NEG_SENTINEL; n_rep];
            let mut l = vec![0.0f32; n_rep];
            let mut acc = vec![0.0f32; n_rep * head_dim];

            let n_blocks = len.div_ceil(block_size);
            for b in 0..n_blocks {
                let blk = block_table[lane * blocks_per_lane + b] as usize;
                let live = block_size.min(len - b * block_size);
                for t in 0..live {
                    let row = ((blk * block_size + t) * num_kv_heads + kv_head) * head_dim;
                    for g in 0..n_rep {
                        let head = kv_head * n_rep + g;
                        let qbase = (lane * num_heads + head) * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[qbase + d] * k_pool[row + d];
                        }
                        let s = dot * scale;
                        // The rescale, and the only three lines that can be wrong in a way the
                        // tensor-op reference would not notice.
                        let m_new = m[g].max(s);
                        let alpha = (m[g] - m_new).exp();
                        let p = (s - m_new).exp();
                        l[g] = l[g] * alpha + p;
                        for d in 0..head_dim {
                            acc[g * head_dim + d] =
                                acc[g * head_dim + d] * alpha + p * v_pool[row + d];
                        }
                        m[g] = m_new;
                    }
                }
            }

            for g in 0..n_rep {
                let head = kv_head * n_rep + g;
                let obase = (lane * num_heads + head) * head_dim;
                // `l >= 1` always: the argmax key contributes `exp(0)`. The clamp is unreachable.
                let inv = 1.0 / l[g].max(1.0e-30);
                for d in 0..head_dim {
                    out[obase + d] = acc[g * head_dim + d] * inv;
                }
            }
        }
    }

    out
}
