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
//! 2. `impl for CubeBackend<R>` is one line and covers CUDA, Metal, Vulkan, wgpu and ROCm at once.
//! 3. `impl for Fusion<B>` is the one the dispatch actually reaches, because `burn::backend::Wgpu`
//!    *is* `Fusion<CubeBackend<WgpuRuntime>>`. It registers the launch as a `CustomOpIr` on the
//!    fusion stream so it is ordered against the `scatter_nd` that wrote this round's KV
//!    immediately before it.

use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{backend_extension, Backend};
use burn::tensor::Device;

// `#[backend_extension]` names each listed backend by its bare type (`<Wgpu as Trait>::…`) as well
// as by its `DispatchTensorKind` variant, so the types have to be in scope here. Each `use` is
// gated exactly as the corresponding entry in the attribute below.
#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "metal")]
use burn::backend::Metal;
#[cfg(feature = "rocm")]
use burn::backend::Rocm;
#[cfg(feature = "vulkan")]
use burn::backend::Vulkan;
#[cfg(feature = "webgpu")]
use burn::backend::WebGpu;
#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;

use burn_cubecl::{CubeBackend, CubeRuntime};

use crate::kernel;

/// Decode-only paged attention as a backend operation.
///
/// PRECONDITION: every check in `crate::paged_decode` passed. Reaching an implementation with an
/// unsupported configuration is a bug in the host gate, not a runtime condition, so the impls
/// assert rather than degrade — with one exception, the pool contiguity check inside
/// `kernel::launch`, which the host cannot see through a `Tensor` and which therefore has to be
/// allowed to decline.
#[backend_extension(
    Cuda:   cfg(feature = "cuda"),
    Rocm:   cfg(feature = "rocm"),
    Metal:  cfg(feature = "metal"),
    Vulkan: cfg(feature = "vulkan"),
    Wgpu:   cfg(feature = "wgpu"),
    WebGpu: cfg(feature = "webgpu"),
)]
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

impl<R: CubeRuntime> PagedDecodeAttention for CubeBackend<R> {
    fn paged_decode(
        q: FloatTensor<Self>,
        k_pool: FloatTensor<Self>,
        v_pool: FloatTensor<Self>,
        block_table: IntTensor<Self>,
        lengths: IntTensor<Self>,
        n_rep: u32,
        scale: f32,
    ) -> FloatTensor<Self> {
        kernel::launch::<R>(
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
/// Is this device one of the backends the trait lists (otherwise the generated dispatch arm is a
/// hard `unimplemented!()`), and does its plane geometry fit the kernel? The second needs the
/// compute client, which is why each arm reaches its own runtime rather than sharing code.
pub(crate) fn device_supports(
    device: &Device,
    n: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> bool {
    #[allow(unused_imports)]
    use burn::backend::DispatchDevice;
    #[allow(unused_imports)]
    use cubecl::Runtime;

    #[allow(unused_mut)]
    let mut supported = false;

    macro_rules! probe {
        ($variant:ident, $runtime:ty) => {
            if let DispatchDevice::$variant(inner) = device.as_dispatch() {
                let client = <$runtime as Runtime>::client(inner);
                supported = kernel::supported::<$runtime>(&client, n, num_kv_heads, head_dim);
            }
        };
    }

    #[cfg(feature = "wgpu")]
    probe!(Wgpu, cubecl::wgpu::WgpuRuntime<cubecl::wgpu::AutoCompiler>);
    #[cfg(feature = "metal")]
    probe!(Metal, cubecl::wgpu::WgpuRuntime<cubecl::wgpu::MslCompiler>);
    #[cfg(feature = "vulkan")]
    probe!(
        Vulkan,
        cubecl::wgpu::WgpuRuntime<cubecl::wgpu::SpirvCompiler>
    );
    #[cfg(feature = "webgpu")]
    probe!(
        WebGpu,
        cubecl::wgpu::WgpuRuntime<cubecl::wgpu::WgslCompiler>
    );
    #[cfg(feature = "cuda")]
    probe!(Cuda, cubecl::cuda::CudaRuntime);
    #[cfg(feature = "rocm")]
    probe!(Rocm, cubecl::hip::HipRuntime);

    supported
}

/// Whether this device runs the kernel when nobody asked for either implementation.
///
/// `BURN_LM_PAGED_ATTENTION` unset means "use whichever is faster here", and the honest answer
/// differs by backend, so it is answered per device rather than by one global default.
///
/// **Metal: yes.** On an M1 Max the kernel beats the tensor-op reference everywhere the roofline
/// bench looks — from 1.2x at the shortest context and one lane to 20x at 4096 tokens across 32
/// lanes — and it reaches about 300 GB/s of roughly 400 GB/s of hardware bandwidth. There is no
/// corner of the grid where it loses, which is what a default needs.
///
/// **Everywhere else: not yet, and CUDA is the interesting case.** An A10G at fp16 serving
/// Llama-3.2-1B at width 16 measured 21.6 / 21.2 / 25.5 ms per decode round at l_max ~100 / ~1000
/// / ~4000 with the kernel, against 30.0 / 52.9 / ~177 ms with the reference — a win at every
/// context length, and at 4000 tokens the reference could not finish the burst at all. So CUDA is
/// not staying opt-in because the kernel is slow there. It is staying opt-in because that is one
/// GPU generation (sm86). sm90 was measured only against the *previous*, unoptimized kernel, where
/// it came out flat, and the change that produced these numbers — splitting a lane's walk across
/// the planes of a cube — is exactly the kind of change whose payoff depends on how many warps the
/// device wanted in the first place. Re-measure an H100 and this becomes a two-line change.
///
/// `BURN_LM_PAGED_ATTENTION=kernel` is how you ask for it in the meantime, and on CUDA today you
/// should.
pub(crate) fn device_defaults_to_kernel(device: &Device) -> bool {
    #[allow(unused_imports)]
    use burn::backend::DispatchDevice;

    #[cfg(feature = "metal")]
    if matches!(device.as_dispatch(), DispatchDevice::Metal(_)) {
        return true;
    }

    let _ = device;
    false
}

/// The fusion implementation — the one that actually runs.
///
/// `burn::backend::Wgpu` is `Fusion<CubeBackend<WgpuRuntime>>`, so a burn-lm tensor never reaches
/// the bare cubecl impl above. Modelled line for line on `burn-vision`'s cube ops.
mod fusion {
    use super::*;

    use burn::backend::Shape;
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
                ) {
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
