use std::ffi::c_void;

use windows::core::{implement, Interface, IUnknown, IUnknownImpl, Ref, Result, BOOL, GUID};
use windows::Win32::Foundation::{E_NOTIMPL, HANDLE, RECT};
use windows::Win32::Graphics::Dxgi::{Common::*, *};

use crate::config::DisplayMode;

pub(super) fn adapter(host: IDXGIAdapter4, mode: DisplayMode) -> IDXGIAdapter {
    let adapter: IDXGIAdapter4 = DisplayAdapter { host, mode }.into();
    adapter.into()
}

#[implement(IDXGIAdapter4)]
struct DisplayAdapter {
    host: IDXGIAdapter4,
    mode: DisplayMode,
}

impl IDXGIObject_Impl for DisplayAdapter_Impl {
    fn SetPrivateData(&self, name: *const GUID, size: u32, data: *const c_void) -> Result<()> {
        unsafe { self.host.SetPrivateData(name, size, data) }
    }

    fn SetPrivateDataInterface(&self, name: *const GUID, unknown: Ref<IUnknown>) -> Result<()> {
        unsafe { self.host.SetPrivateDataInterface(name, unknown.as_ref()) }
    }

    fn GetPrivateData(&self, name: *const GUID, size: *mut u32, data: *mut c_void) -> Result<()> {
        unsafe { self.host.GetPrivateData(name, size, data) }
    }

    fn GetParent(&self, iid: *const GUID, out: *mut *mut c_void) -> Result<()> {
        let host: &IDXGIObject = &self.host;
        unsafe { (host.vtable().GetParent)(host.as_raw(), iid, out).ok() }
    }
}

impl IDXGIAdapter_Impl for DisplayAdapter_Impl {
    fn EnumOutputs(&self, index: u32) -> Result<IDXGIOutput> {
        if index != 0 {
            return Err(DXGI_ERROR_NOT_FOUND.into());
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        let mode = self.mode;
        ONCE.call_once(|| crate::xlog_info!("D3D12.X: reported console display {}x{} @ {} Hz SDR", mode.width, mode.height, mode.hz));
        let output: IDXGIOutput6 = DisplayOutput { parent: self.to_interface::<IDXGIAdapter4>().into(), mode }.into();
        Ok(output.into())
    }

    fn GetDesc(&self) -> Result<DXGI_ADAPTER_DESC> {
        unsafe { self.host.GetDesc() }
    }

    fn CheckInterfaceSupport(&self, iid: *const GUID) -> Result<i64> {
        unsafe { self.host.CheckInterfaceSupport(iid) }
    }
}

impl IDXGIAdapter1_Impl for DisplayAdapter_Impl {
    fn GetDesc1(&self) -> Result<DXGI_ADAPTER_DESC1> {
        unsafe { self.host.GetDesc1() }
    }
}

impl IDXGIAdapter2_Impl for DisplayAdapter_Impl {
    fn GetDesc2(&self) -> Result<DXGI_ADAPTER_DESC2> {
        unsafe { self.host.GetDesc2() }
    }
}

impl IDXGIAdapter3_Impl for DisplayAdapter_Impl {
    fn RegisterHardwareContentProtectionTeardownStatusEvent(&self, event: HANDLE) -> Result<u32> {
        unsafe { self.host.RegisterHardwareContentProtectionTeardownStatusEvent(event) }
    }

    fn UnregisterHardwareContentProtectionTeardownStatus(&self, cookie: u32) {
        unsafe { self.host.UnregisterHardwareContentProtectionTeardownStatus(cookie) }
    }

    fn QueryVideoMemoryInfo(&self, node: u32, group: DXGI_MEMORY_SEGMENT_GROUP, info: *mut DXGI_QUERY_VIDEO_MEMORY_INFO) -> Result<()> {
        unsafe { self.host.QueryVideoMemoryInfo(node, group, info) }
    }

    fn SetVideoMemoryReservation(&self, node: u32, group: DXGI_MEMORY_SEGMENT_GROUP, reservation: u64) -> Result<()> {
        unsafe { self.host.SetVideoMemoryReservation(node, group, reservation) }
    }

    fn RegisterVideoMemoryBudgetChangeNotificationEvent(&self, event: HANDLE) -> Result<u32> {
        unsafe { self.host.RegisterVideoMemoryBudgetChangeNotificationEvent(event) }
    }

    fn UnregisterVideoMemoryBudgetChangeNotification(&self, cookie: u32) {
        unsafe { self.host.UnregisterVideoMemoryBudgetChangeNotification(cookie) }
    }
}

impl IDXGIAdapter4_Impl for DisplayAdapter_Impl {
    fn GetDesc3(&self) -> Result<DXGI_ADAPTER_DESC3> {
        unsafe { self.host.GetDesc3() }
    }
}

#[implement(IDXGIOutput6)]
struct DisplayOutput {
    parent: IUnknown,
    mode: DisplayMode,
}

impl DisplayOutput {
    fn mode(&self, format: DXGI_FORMAT) -> Option<DXGI_MODE_DESC> {
        match format {
            DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
            | DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
            | DXGI_FORMAT_R10G10B10A2_UNORM | DXGI_FORMAT_R16G16B16A16_FLOAT => Some(DXGI_MODE_DESC {
                Width: self.mode.width,
                Height: self.mode.height,
                RefreshRate: DXGI_RATIONAL { Numerator: self.mode.hz, Denominator: 1 },
                Format: format,
                ScanlineOrdering: DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE,
                Scaling: DXGI_MODE_SCALING_CENTERED,
            }),
            _ => None,
        }
    }

    fn mode1(&self, format: DXGI_FORMAT) -> Option<DXGI_MODE_DESC1> {
        self.mode(format).map(|m| DXGI_MODE_DESC1 {
            Width: m.Width,
            Height: m.Height,
            RefreshRate: m.RefreshRate,
            Format: m.Format,
            ScanlineOrdering: m.ScanlineOrdering,
            Scaling: m.Scaling,
            Stereo: false.into(),
        })
    }
}

unsafe fn write_modes<T>(count: *mut u32, out: *mut T, mode: Option<T>) -> Result<()> {
    if count.is_null() {
        return Err(DXGI_ERROR_INVALID_CALL.into());
    }
    let Some(mode) = mode else {
        count.write(0);
        return Ok(());
    };
    if out.is_null() {
        count.write(1);
        return Ok(());
    }
    let capacity = count.replace(1);
    if capacity == 0 {
        return Err(DXGI_ERROR_MORE_DATA.into());
    }
    out.write(mode);
    Ok(())
}

impl IDXGIObject_Impl for DisplayOutput_Impl {
    fn SetPrivateData(&self, _: *const GUID, _: u32, _: *const c_void) -> Result<()> {
        Err(E_NOTIMPL.into())
    }

    fn SetPrivateDataInterface(&self, _: *const GUID, _: Ref<IUnknown>) -> Result<()> {
        Err(E_NOTIMPL.into())
    }

    fn GetPrivateData(&self, _: *const GUID, _: *mut u32, _: *mut c_void) -> Result<()> {
        Err(DXGI_ERROR_NOT_FOUND.into())
    }

    fn GetParent(&self, iid: *const GUID, out: *mut *mut c_void) -> Result<()> {
        unsafe { self.parent.query(iid, out).ok() }
    }
}

impl IDXGIOutput_Impl for DisplayOutput_Impl {
    fn GetDesc(&self) -> Result<DXGI_OUTPUT_DESC> {
        Ok(DXGI_OUTPUT_DESC {
            DesktopCoordinates: RECT { left: 0, top: 0, right: self.mode.width as i32, bottom: self.mode.height as i32 },
            AttachedToDesktop: true.into(),
            Rotation: DXGI_MODE_ROTATION_IDENTITY,
            ..Default::default()
        })
    }

    fn GetDisplayModeList(&self, format: DXGI_FORMAT, _: DXGI_ENUM_MODES, count: *mut u32, out: *mut DXGI_MODE_DESC) -> Result<()> {
        unsafe { write_modes(count, out, self.mode(format)) }
    }

    fn FindClosestMatchingMode(&self, requested: *const DXGI_MODE_DESC, out: *mut DXGI_MODE_DESC, device: Ref<IUnknown>) -> Result<()> {
        unsafe {
            let format = requested.as_ref().ok_or(DXGI_ERROR_INVALID_CALL)?.Format;
            if out.is_null() {
                return Err(DXGI_ERROR_INVALID_CALL.into());
            }
            out.write(self.mode(closest_format(format, device)?).ok_or(DXGI_ERROR_NOT_FOUND)?);
        }
        Ok(())
    }

    fn WaitForVBlank(&self) -> Result<()> { Err(E_NOTIMPL.into()) }
    fn TakeOwnership(&self, _: Ref<IUnknown>, _: BOOL) -> Result<()> { Err(E_NOTIMPL.into()) }
    fn ReleaseOwnership(&self) {}
    fn GetGammaControlCapabilities(&self, _: *mut DXGI_GAMMA_CONTROL_CAPABILITIES) -> Result<()> { Err(E_NOTIMPL.into()) }
    fn SetGammaControl(&self, _: *const DXGI_GAMMA_CONTROL) -> Result<()> { Err(E_NOTIMPL.into()) }
    fn GetGammaControl(&self, _: *mut DXGI_GAMMA_CONTROL) -> Result<()> { Err(E_NOTIMPL.into()) }
    fn SetDisplaySurface(&self, _: Ref<IDXGISurface>) -> Result<()> { Err(E_NOTIMPL.into()) }
    fn GetDisplaySurfaceData(&self, _: Ref<IDXGISurface>) -> Result<()> { Err(E_NOTIMPL.into()) }
    fn GetFrameStatistics(&self, _: *mut DXGI_FRAME_STATISTICS) -> Result<()> { Err(E_NOTIMPL.into()) }
}

fn closest_format(format: DXGI_FORMAT, device: Ref<IUnknown>) -> Result<DXGI_FORMAT> {
    if format != DXGI_FORMAT_UNKNOWN {
        Ok(format)
    } else if device.as_ref().is_some() {
        Ok(DXGI_FORMAT_R8G8B8A8_UNORM)
    } else {
        Err(DXGI_ERROR_INVALID_CALL.into())
    }
}

impl IDXGIOutput1_Impl for DisplayOutput_Impl {
    fn GetDisplayModeList1(&self, format: DXGI_FORMAT, _: DXGI_ENUM_MODES, count: *mut u32, out: *mut DXGI_MODE_DESC1) -> Result<()> {
        unsafe { write_modes(count, out, self.mode1(format)) }
    }

    fn FindClosestMatchingMode1(&self, requested: *const DXGI_MODE_DESC1, out: *mut DXGI_MODE_DESC1, device: Ref<IUnknown>) -> Result<()> {
        unsafe {
            let requested = requested.as_ref().ok_or(DXGI_ERROR_INVALID_CALL)?;
            if requested.Stereo.as_bool() {
                return Err(DXGI_ERROR_NOT_FOUND.into());
            }
            if out.is_null() {
                return Err(DXGI_ERROR_INVALID_CALL.into());
            }
            out.write(self.mode1(closest_format(requested.Format, device)?).ok_or(DXGI_ERROR_NOT_FOUND)?);
        }
        Ok(())
    }

    fn GetDisplaySurfaceData1(&self, _: Ref<IDXGIResource>) -> Result<()> { Err(E_NOTIMPL.into()) }
    fn DuplicateOutput(&self, _: Ref<IUnknown>) -> Result<IDXGIOutputDuplication> { Err(E_NOTIMPL.into()) }
}

impl IDXGIOutput2_Impl for DisplayOutput_Impl {
    fn SupportsOverlays(&self) -> BOOL { false.into() }
}

impl IDXGIOutput3_Impl for DisplayOutput_Impl {
    fn CheckOverlaySupport(&self, _: DXGI_FORMAT, _: Ref<IUnknown>) -> Result<u32> { Ok(0) }
}

impl IDXGIOutput4_Impl for DisplayOutput_Impl {
    fn CheckOverlayColorSpaceSupport(&self, _: DXGI_FORMAT, _: DXGI_COLOR_SPACE_TYPE, _: Ref<IUnknown>) -> Result<u32> { Ok(0) }
}

impl IDXGIOutput5_Impl for DisplayOutput_Impl {
    fn DuplicateOutput1(&self, _: Ref<IUnknown>, _: u32, _: u32, _: *const DXGI_FORMAT) -> Result<IDXGIOutputDuplication> { Err(E_NOTIMPL.into()) }
}

impl IDXGIOutput6_Impl for DisplayOutput_Impl {
    fn GetDesc1(&self) -> Result<DXGI_OUTPUT_DESC1> {
        let desc = IDXGIOutput_Impl::GetDesc(self)?;
        Ok(DXGI_OUTPUT_DESC1 {
            DesktopCoordinates: desc.DesktopCoordinates,
            AttachedToDesktop: desc.AttachedToDesktop,
            Rotation: desc.Rotation,
            BitsPerColor: 8,
            ColorSpace: DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
            RedPrimary: [0.640, 0.330],
            GreenPrimary: [0.300, 0.600],
            BluePrimary: [0.150, 0.060],
            WhitePoint: [0.3127, 0.3290],
            MaxLuminance: 100.0,
            MaxFullFrameLuminance: 100.0,
            ..Default::default()
        })
    }

    fn CheckHardwareCompositionSupport(&self) -> Result<u32> { Ok(0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reported_display_dxgi_methods_agree() {
        unsafe {
            let factory: IDXGIFactory1 = CreateDXGIFactory1().unwrap();
            let output: IDXGIOutput6 = DisplayOutput {
                parent: factory.into(),
                mode: DisplayMode { width: 1280, height: 720, hz: 60 },
            }.into();
            let base = output.cast::<IDXGIOutput>().unwrap();
            let desc = base.GetDesc().unwrap();
            assert_eq!((desc.DesktopCoordinates.right, desc.DesktopCoordinates.bottom), (1280, 720));
            let desc1 = output.GetDesc1().unwrap();
            assert_eq!(desc1.DesktopCoordinates, desc.DesktopCoordinates);
            assert_eq!(desc1.ColorSpace, DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);
            assert_eq!(desc1.BitsPerColor, 8);

            let format = DXGI_FORMAT_R8G8B8A8_UNORM;
            let flags = DXGI_ENUM_MODES(0);
            let mut count = 0;
            let mut mode = DXGI_MODE_DESC::default();
            let err = base.GetDisplayModeList(format, flags, &mut count, Some(&mut mode)).unwrap_err();
            assert_eq!((err.code(), count), (DXGI_ERROR_MORE_DATA, 1));
            base.GetDisplayModeList(format, flags, &mut count, Some(&mut mode)).unwrap();
            assert_eq!((mode.Width, mode.Height, mode.RefreshRate.Numerator, mode.RefreshRate.Denominator), (1280, 720, 60, 1));

            let modern = output.cast::<IDXGIOutput1>().unwrap();
            count = 0;
            modern.GetDisplayModeList1(format, flags, &mut count, None).unwrap();
            assert_eq!(count, 1);
            let mut mode1 = DXGI_MODE_DESC1::default();
            modern.GetDisplayModeList1(format, flags, &mut count, Some(&mut mode1)).unwrap();
            assert_eq!((mode1.Width, mode1.Height, mode1.RefreshRate, mode1.Format), (mode.Width, mode.Height, mode.RefreshRate, mode.Format));
            assert!(!mode1.Stereo.as_bool());
            let mut closest = DXGI_MODE_DESC1::default();
            modern.FindClosestMatchingMode1(&DXGI_MODE_DESC1 { Width: 3840, Height: 2160, Format: format, ..Default::default() }, &mut closest, None::<&IUnknown>).unwrap();
            assert_eq!(closest, mode1);
        }
    }
}
