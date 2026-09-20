//! Getting the kernel from a `Tensor` to a concrete cubecl backend, and back.
//!
//! Three layers, and the middle one is where the week goes:
//!
//! 1. [`PagedDecodeAttention`] is a backend-extension trait. `#[backend_extension]` generates
//!    `impl PagedDecodeAttention for Dispatch`, which matches the incoming tensors' runtime backend
//!    tag and forwards to that backend's own impl. Backends it was not told about hit a hard
//!    `unimplemented!()` in the generated arm — a trait *default body* does not fill those in — so
//!    the "can we?" question is answered on the host, before this trait is ever reached (see
//!    [`device_supports`] and `crate::paged_decode`).
//! 2. `impl for CubeBackend` is one line and covers CUDA, Metal, Vulkan, wgpu and ROCm at once —
//!    since cubecl's runtime erasure it does so *literally*, because those runtimes are no longer
//!    separate types: `CubeBackend` takes no runtime parameter and a tensor's runtime is what its
//!    device says.
//! 3. `impl for Fusion<B>` is the one the dispatch actually reaches, because `burn::backend::Cube`
//!    — which is what the generated arm names — *is* `Fusion<CubeBackend>`. It registers the launch
//!    as a `CustomOpIr` on the fusion stream so it is ordered against the `scatter_nd` that wrote
//!    this round's KV immediately before it.

use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{backend_extension, Backend};
use burn::tensor::Device;

use burn_cubecl::CubeBackend;

use crate::kernel;

/// Decode-only paged attention as a backend operation.
///
/// PRECONDITION: every check in `crate::paged_decode` passed. Reaching an implementation with an
/// unsupported configuration is a bug in the host gate, not a runtime condition, so the impls
/// assert rather than degrade. There is no exception: `paged_decode` returns `FloatTensor`, and
/// under fusion this runs from inside the registered operation, which has to produce one. The pool
/// contiguity that `kernel::launch` needs but cannot see through a `Tensor` is answered up here
/// instead, by asking the runtime whether it would pitch a row of `head_dim` at all — see
/// `kernel::supported`.
// One entry, not six. The macro's backend list names *selectors* — `Cube`, `Flex`, `NdArray`,
// `LibTorch`, `Remote` — and since cubecl's runtime erasure `Cube` is every cubecl runtime at once:
// CUDA, ROCm, Metal, Vulkan, wgpu, WebGPU and the CPU runtime. `Cuda` and `Wgpu` are no longer
// spellings the macro accepts, which is why this reads as a loss of CUDA and is not one; what used
// to be six arms dispatching on the tensor's backend *type* is now one arm, with the runtime
// carried by the device value inside it.
#[backend_extension(Cube: cfg(feature = "cube-backend"))]
pub trait PagedDecodeAttention: Backend {
    /// Attend one query position per lane over the two pools, addressed by `block_table` and
    /// bounded by `lengths`. Returns `[n, num_heads, 1, head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn paged_decode(
        q: FloatTensor<Self>,
        k_pool: FloatTensor<Self>,
        v_pool: FloatTensor<Self>,
        block_table: IntTensor<Self>,
        lengths: IntTensor<Self>,
        n_rep: u32,
        scale: f32,
    ) -> FloatTensor<Self>;
}

impl PagedDecodeAttention for CubeBackend {
    fn paged_decode(
        q: FloatTensor<Self>,
        k_pool: FloatTensor<Self>,
        v_pool: FloatTensor<Self>,
        block_table: IntTensor<Self>,
        lengths: IntTensor<Self>,
        n_rep: u32,
        scale: f32,
    ) -> FloatTensor<Self> {
        kernel::launch(
            q,
            k_pool,
            v_pool,
            block_table,
            lengths,
            n_rep as usize,
            scale,
        )
        .expect(
            "paged_decode reached the kernel on a configuration it cannot serve; the host \
             availability gate and the launch's own preconditions have drifted apart",
        )
    }
}

/// Whether the kernel is reachable for `device` at this problem size.
///
/// Two questions in one, both of which have to be answered before the extension trait is called.
/// Is this device a cubecl device at all (otherwise the generated dispatch arm is a hard
/// `unimplemented!()`), and does its plane geometry fit the kernel? The second needs the compute
/// client, which used to mean one probe per runtime, each naming its own `Runtime` type to reach
/// `Runtime::client`. Runtime erasure removed both halves of that: the device *is* the runtime
/// tag, and `CubeDevice::client` hands back the one erased `Client` whatever it names.
#[allow(unused_variables)]
pub(crate) fn device_supports(
    device: &Device,
    n: usize,
    num_kv_heads: usize,
    head_dim: usize,
    elem_size: usize,
) -> bool {
    #[cfg(feature = "cube-backend")]
    if let burn::backend::DispatchDevice::Cube(cube) = device.as_dispatch() {
        return kernel::supported(&cube.client(), n, num_kv_heads, head_dim, elem_size);
    }

    false
}

/// Whether this device runs the kernel when nobody asked for either implementation.
///
/// `BURN_LM_PAGED_ATTENTION` unset means "use whichever is faster here", and the honest answer
/// differs by backend, so it is answered per device rather than by one global default.
///
/// **Metal and CUDA: yes.** On both, the kernel beat the tensor-op reference at every point
/// measured. An M1 Max runs from 1.2x at the shortest context and one lane to 20x at 4096 tokens
/// across 32 lanes, reaching about 300 GB/s of roughly 400 GB/s of hardware bandwidth. An A10G at
/// fp16, serving Llama-3.2-1B at width 16, measured 21.6 / 21.2 / 25.5 ms per decode round at
/// l_max ~100 / ~1000 / ~4000 against 30.0 / 52.9 / ~177 ms for the reference — and at 4000 tokens
/// the reference could not finish the burst at all.
///
/// The one CUDA generation not re-measured since the kernel was optimized is sm90, where the
/// *previous* kernel came out roughly flat (1.13x). That is not a reason to withhold the default,
/// because the change since then splits a lane's block walk across the planes of a cube, and it
/// pays in proportion to how starved the device was for parallelism. The old grid was
/// `num_kv_heads x lanes` — 128 cubes for a batch of 16 — and an H100 has more streaming
/// multiprocessors to feed than an A10G, so it was the *more* starved of the two. The mechanism
/// predicts a larger gain there, not a smaller one. A device that turns out to disagree can say so
/// with `BURN_LM_PAGED_ATTENTION=reference`, and this function should grow a case for it.
///
/// **wgpu, Vulkan, WebGPU, ROCm, cubecl's native Metal runtime: not yet**, and the reason is
/// different in kind. Those are not merely unmeasured hardware but unmeasured *compilation paths* —
/// cubecl generates WGSL, SPIR-V, HIP or its own Metal output rather than the MSL-through-wgpu and
/// PTX the numbers above were taken on, so neither transfers. They stay opt-in until somebody runs
/// the roofline bench on one.
///
/// # Asking the device which path it is
///
/// This used to be a match on `DispatchDevice::Metal` / `DispatchDevice::Cuda`. Those variants are
/// gone with runtime erasure: every cubecl device is `DispatchDevice::Cube`, and what it names is
/// inside. CUDA is still a variant of its own, so that half is unchanged in substance.
///
/// "Metal" is the awkward half, because burn's `metal` feature is not a runtime — it is wgpu with
/// its MSL compiler, and the wgpu device only says which graphics API it is pinned to, which for
/// `Device::wgpu` is `Auto`. So an unpinned device falls back to `cfg!(feature = "metal")`. That is
/// not a new approximation: the old `DispatchDevice::Metal` variant was itself produced by a
/// cargo-feature priority chain in burn's `From<WgpuDevice>` (metal, then vulkan, then webgpu), so
/// the question was already being answered by the build's features and never by the device. A build
/// that enables both `metal` and `vulkan` gets the Metal answer either way.
pub(crate) fn device_defaults_to_kernel(device: &Device) -> bool {
    #[cfg(feature = "cube-backend")]
    {
        use burn::backend::{CubeDevice, DispatchDevice};

        let DispatchDevice::Cube(cube) = device.as_dispatch() else {
            return false;
        };

        return match cube {
            CubeDevice::Cuda(_) => true,
            #[cfg(feature = "wgpu")]
            CubeDevice::Wgpu(wgpu) => {
                matches!(wgpu.backend, burn::backend::WgpuBackend::Metal) || cfg!(feature = "metal")
            }
            _ => false,
        };
    }

    #[cfg(not(feature = "cube-backend"))]
    {
        let _ = device;
        false
    }
}

/// The fusion implementation — the one that actually runs.
///
/// `burn::backend::Cube` — the type the generated dispatch arm names, and what `burn::backend::Wgpu`
/// and `burn::backend::Cuda` are both aliases of now — is `Fusion<CubeBackend>`, so a burn-lm tensor
/// never reaches the bare cubecl impl above. Modelled line for line on `burn-vision`'s cube ops.
mod fusion {
    use super::*;

    use burn::backend::{ExecutionError, Shape};
    use burn_fusion::{
        stream::{Operation, StreamId},
        Fusion, FusionBackend, FusionRuntime,
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, ScalarIr, TensorIr};

    impl<B: FusionBackend + PagedDecodeAttention> PagedDecodeAttention for Fusion<B> {
        fn paged_decode(
            q: FloatTensor<Self>,
            k_pool: FloatTensor<Self>,
            v_pool: FloatTensor<Self>,
            block_table: IntTensor<Self>,
            lengths: IntTensor<Self>,
            n_rep: u32,
            scale: f32,
        ) -> FloatTensor<Self> {
            #[derive(derive_new::new, Clone, Debug)]
            struct PagedDecode<B> {
                desc: CustomOpIr,
                n_rep: u32,
                scale: f32,
                _b: core::marker::PhantomData<B>,
            }

            impl<B1: FusionBackend + PagedDecodeAttention> Operation<B1::FusionRuntime> for PagedDecode<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) -> Result<(), ExecutionError> {
                    let ([q, kp, vp, bt, ln], [out]) = self.desc.as_fixed::<5, 1>();
                    let result = B1::paged_decode(
                        handles.get_float_tensor::<B1>(q),
                        handles.get_float_tensor::<B1>(kp),
                        handles.get_float_tensor::<B1>(vp),
                        handles.get_int_tensor::<B1>(bt),
                        handles.get_int_tensor::<B1>(ln),
                        self.n_rep,
                        self.scale,
                    );
                    handles.register_float_tensor::<B1>(&out.id, result);
                    // Always `Ok`. `execute` grew a `Result` so that a failing operation claims only
                    // its own writes instead of poisoning the whole stream (burn#5535), but nothing
                    // below this line reports failure that way: every condition the kernel cannot
                    // serve was already answered `None` on the host, and what is left — the
                    // gate and the launch's preconditions having drifted apart — is a bug that
                    // should abort loudly rather than be handed back as a recoverable error.
                    Ok(())
                }
            }

            let client = q.client.clone();
            let streams = StreamId::current();
            let [n, num_heads, _one, head_dim] = q.shape.dims::<4>();
            let out = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([n, num_heads, 1, head_dim]),
                q.dtype,
            );

            // MANDATORY, and the single most dangerous line in this file. `FusionTensor::into_ir`
            // marks a handle `ReadWrite` when its refcount is 1, and a `ReadWrite` pool input
            // invites fusion to hand this kernel the POOL as a writable in-place target: silent
            // KV-cache corruption. Cloning first forces the count above 1, hence `ReadOnly`. This
            // is the same reason `BlockStore::gather` clones the pool handle rather than moving it,
            // and it is gated by `pool_bytes_are_unchanged_by_a_kernel_decode` in
            // burn-lm-paged-kv. Do not "simplify" these clones away.
            let inputs = [
                q.into_ir(),
                k_pool.clone().into_ir(),
                v_pool.clone().into_ir(),
                block_table.clone().into_ir(),
                lengths.clone().into_ir(),
            ];

            // `with_scalars`, not `new`: scalars are carried through fusion's relativization, so a
            // cached graph replays with a fresh `scale` instead of baking in the first one it saw.
            let desc = CustomOpIr::with_scalars(
                "burn_lm::paged_decode",
                &inputs,
                &[out],
                vec![ScalarIr::UInt(n_rep as u64), ScalarIr::Float(scale as f64)],
            );

            client
                .register(
                    streams,
                    OperationIr::Custom(desc.clone()),
                    PagedDecode::<B>::new(desc, n_rep, scale),
                )
                .output()
        }
    }
}
