//! One place every test in this crate gets its device from, and the reason there is only one.
//!
//! Two cubecl devices in one multithreaded test binary is a hazard, and it is not a subtle one:
//! bringing a second device's channel up while other threads are already issuing work on the first
//! spins inside `cubecl_common::device::handle::channel::ChannelDeviceState::init` — on this
//! machine, at `cargo test`'s default thread count, that is eight threads at 100% CPU making no
//! progress at all. Four threads or fewer are fine, which is exactly the shape of a contention bug
//! rather than a deadlock.
//!
//! The f16 half of the kernel grid genuinely needs a second device: a device's float dtype is
//! registry state keyed by dispatch variant and can only be set before that variant is first used,
//! so f32 and f16 pools cannot share one. The fix is therefore not to avoid the second device but
//! to bring *both* up inside a single `OnceLock`, before any test has started work on either —
//! whoever asks first initializes the pair while everybody else waits on the lock, and by the time
//! any test owns a device both channels are already live.
//!
//! Every test in this crate goes through here, including the ones that have nothing to do with the
//! kernel. That is the point: it only takes one test reaching for `Device::default()` on its own to
//! put the binary back in the racing state.

use std::sync::OnceLock;

use burn::tensor::{Device, Tensor};

/// The devices this crate's tests run on, initialized together.
struct TestDevices {
    /// The build's default device, at its default (f32) dtype. Everything but the f16 grid.
    default: Device,
    /// A second dispatch variant, configured to f16 storage — see the module docs.
    #[cfg(all(feature = "metal", feature = "wgpu"))]
    f16: Device,
}

/// Force a device's channel all the way up, so that no later thread has to.
///
/// Constructing a `Device` is cheap and lazy; the channel comes up on first use. A one-element
/// tensor read back to the host is the smallest thing that guarantees "first use" has happened.
fn warm(device: &Device) {
    let _ = Tensor::<1>::zeros([1], device).into_data();
}

fn devices() -> &'static TestDevices {
    static DEVICES: OnceLock<TestDevices> = OnceLock::new();
    DEVICES.get_or_init(|| {
        #[cfg(all(feature = "metal", feature = "wgpu"))]
        let f16 = {
            use burn::tensor::{f16, DeviceConfig, Element};
            // `Device::wgpu` rather than the default variant: the default is what every other test
            // uses, and its dtype is therefore already locked in by the time anyone asks for f16.
            let mut device = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
            device
                .configure(DeviceConfig::default().float_dtype(f16::dtype()))
                .expect(
                    "the f16 device must be configured before anything initializes it: something \
                     in this test binary now uses Device::wgpu outside this module",
                );
            warm(&device);
            device
        };

        let default = Device::default();
        warm(&default);

        TestDevices {
            default,
            #[cfg(all(feature = "metal", feature = "wgpu"))]
            f16,
        }
    })
}

/// The device for everything that is not the f16 grid: the build's default, at f32.
pub(crate) fn test_device() -> Device {
    devices().default.clone()
}

/// The f16-storage device the kernel's f16 grid runs on.
#[cfg(all(feature = "metal", feature = "wgpu"))]
pub(crate) fn f16_device() -> Device {
    devices().f16.clone()
}
