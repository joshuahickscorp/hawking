//! Optional energy for `pJ_per_weight_served`.
//!
//! `powermetrics` requires root on this machine (`sudo -n` fails). IOReport's
//! `Energy Model` group is readable **without root** via `libIOReport.dylib`
//! (dyld shared cache). This session verified:
//!
//! - `GPU Energy` (nJ) increments in real time (~1 W idle with a display).
//! - `GPU0_0`, `DRAM0_0`, `DRAM0_1`, `DIE_*_CPU Energy` exist as mJ channels
//!   but `IOReportSimpleGetIntegerValue` did not increment them over a 1 s
//!   idle window. They are listed, not treated as a working token joule.
//!
//! An idle sample is not `joules_per_token`. Only a wrap around the same
//! interval as TOKEN_NS, or a caller-supplied joule, may fill pJ.

use serde::{Deserialize, Serialize};

/// Approximate DRAM round-trip for one isolated weight. Not a measurement.
pub const SINGLE_WEIGHT_DRAM_ROUND_TRIP_NS: f64 = 100.0;

/// Command a human runs later to fill energy without re-deriving geometry.
pub const ENERGY_FILL_COMMAND: &str = "sudo powermetrics --samplers gpu_power,cpu_power -i 50 -n 40 --show-process-coalition -o receipts/ascent-2026-08-16/POWERMETRICS_<lane>.txt";

pub const ENERGY_FILL_HOWTO: &str = "Run the timed token under the GPU lock and, in another terminal as root, start powermetrics before the token body (not during init). joules_per_token ≈ (mean GPU+CPU watts) × (token_ns × 1e-9). Then re-emit TOKEN_NS with EmitMeta.joules_per_token set. Do not copy a datasheet TDP. Prefer wrapping EnergySampler::start/stop around the same interval as body_ns / decode wall_ns. IOReport Energy Model `GPU Energy` (nJ) is the verified non-root GPU rail; it is GPU-only, not DRAM.";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnergyReport {
    pub energy_available: bool,
    pub reason: String,
    pub joules_per_token: Option<f64>,
    pub scope: EnergyScope,
    pub non_root_ioreport: IoreportFinding,
    pub fill_command: String,
    pub fill_howto: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyScope {
    None,
    CallerJoulesPerToken,
    IoreportGpuEnergyNj,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IoreportFinding {
    pub probed: bool,
    pub libioreport_dlopen_without_root: bool,
    pub energy_model_channels: Option<u32>,
    pub gpu_energy_nj_increments_without_root: Option<bool>,
    pub dram_mj_channels_present: bool,
    pub dram_mj_observed_incrementing: Option<bool>,
    pub note: String,
}

impl IoreportFinding {
    pub fn documented() -> Self {
        Self {
            probed: true,
            libioreport_dlopen_without_root: true,
            energy_model_channels: Some(565),
            gpu_energy_nj_increments_without_root: Some(true),
            dram_mj_channels_present: true,
            dram_mj_observed_incrementing: Some(false),
            note: "2026-08-16 this machine, no root. IOReportCopyChannelsInGroup(\"Energy Model\") returns 565 channels. GPU Energy (nJ) incremented ~0.98 W over a 1 s idle window. GPU0_0 / DRAM0_0 / DRAM0_1 / DIE_*_CPU Energy (mJ) were present but SimpleGetIntegerValue did not increment over 1 s. powermetrics still requires root.".into(),
        }
    }

    pub fn not_probed() -> Self {
        Self {
            probed: false,
            libioreport_dlopen_without_root: true,
            energy_model_channels: Some(565),
            gpu_energy_nj_increments_without_root: Some(true),
            dram_mj_channels_present: true,
            dram_mj_observed_incrementing: Some(false),
            note: "Standing finding from frontier-fs-per-weight. Call probe_energy_model() to refresh.".into(),
        }
    }
}

impl EnergyReport {
    pub fn unavailable(reason: &str) -> Self {
        Self {
            energy_available: false,
            reason: reason.to_owned(),
            joules_per_token: None,
            scope: EnergyScope::None,
            non_root_ioreport: IoreportFinding::documented(),
            fill_command: ENERGY_FILL_COMMAND.to_owned(),
            fill_howto: ENERGY_FILL_HOWTO.to_owned(),
        }
    }

    pub fn from_caller_joules(joules_per_token: f64) -> Self {
        Self {
            energy_available: true,
            reason: "caller supplied joules_per_token".into(),
            joules_per_token: Some(joules_per_token),
            scope: EnergyScope::CallerJoulesPerToken,
            non_root_ioreport: IoreportFinding::documented(),
            fill_command: ENERGY_FILL_COMMAND.to_owned(),
            fill_howto: ENERGY_FILL_HOWTO.to_owned(),
        }
    }

    pub fn from_ioreport_gpu_nj(delta_nj: u64, window_s: f64) -> Self {
        let joules = delta_nj as f64 / 1.0e9;
        Self {
            energy_available: true,
            reason: format!(
                "IOReport Energy Model GPU Energy Δ={delta_nj} nJ over {window_s:.4}s (GPU rail only, not DRAM; DIRTY if other lanes ran)"
            ),
            joules_per_token: Some(joules),
            scope: EnergyScope::IoreportGpuEnergyNj,
            non_root_ioreport: IoreportFinding::documented(),
            fill_command: ENERGY_FILL_COMMAND.to_owned(),
            fill_howto: ENERGY_FILL_HOWTO.to_owned(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnergyProbeReport {
    pub root_available: bool,
    pub powermetrics_without_root: String,
    pub ioreport: IoreportFinding,
    pub sample: Option<EnergyWindowSample>,
    pub verdict: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnergyWindowSample {
    pub window_s: f64,
    pub gpu_energy_nj_t0: u64,
    pub gpu_energy_nj_t1: u64,
    pub gpu_energy_delta_nj: u64,
    pub gpu_watts_over_window: f64,
    pub label: String,
    pub not_joules_per_token: bool,
}

/// Two IOReport snapshots. `stop` subtracts `GPU Energy` (nJ).
pub struct EnergySampler {
    inner: Option<SamplerInner>,
}

struct SamplerInner {
    t0_gpu_nj: u64,
    started: std::time::Instant,
}

impl EnergySampler {
    /// Open Energy Model and take t0. Never panics; returns a dead sampler
    /// if IOReport is missing (non-macOS or dlopen fail).
    pub fn start() -> Self {
        match read_gpu_energy_nj() {
            Ok(t0) => Self {
                inner: Some(SamplerInner {
                    t0_gpu_nj: t0,
                    started: std::time::Instant::now(),
                }),
            },
            Err(_) => Self { inner: None },
        }
    }

    /// GPU-rail joules over the interval, if both snapshots worked.
    pub fn stop(self) -> Option<(f64, EnergyWindowSample)> {
        let inner = self.inner?;
        let t1 = read_gpu_energy_nj().ok()?;
        let dt = inner.started.elapsed().as_secs_f64().max(1e-9);
        let d = t1.saturating_sub(inner.t0_gpu_nj);
        let sample = EnergyWindowSample {
            window_s: dt,
            gpu_energy_nj_t0: inner.t0_gpu_nj,
            gpu_energy_nj_t1: t1,
            gpu_energy_delta_nj: d,
            gpu_watts_over_window: (d as f64 / 1.0e9) / dt,
            label: "DIRTY_ENGINEERING".into(),
            not_joules_per_token: true,
        };
        Some((d as f64 / 1.0e9, sample))
    }
}

/// Live probe: dlopen, channel count, 200 ms GPU Energy increment.
pub fn probe_energy_model() -> EnergyProbeReport {
    let mut finding = IoreportFinding::documented();
    let sample = match read_gpu_energy_nj() {
        Ok(t0) => {
            finding.libioreport_dlopen_without_root = true;
            if let Ok(n) = energy_model_channel_count() {
                finding.energy_model_channels = Some(n);
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
            match read_gpu_energy_nj() {
                Ok(t1) => {
                    let d = t1.saturating_sub(t0);
                    finding.gpu_energy_nj_increments_without_root = Some(d > 0);
                    Some(EnergyWindowSample {
                        window_s: 0.2,
                        gpu_energy_nj_t0: t0,
                        gpu_energy_nj_t1: t1,
                        gpu_energy_delta_nj: d,
                        gpu_watts_over_window: (d as f64 / 1.0e9) / 0.2,
                        label: "DIRTY_ENGINEERING".into(),
                        not_joules_per_token: true,
                    })
                }
                Err(_) => None,
            }
        }
        Err(e) => {
            finding.libioreport_dlopen_without_root = false;
            finding.note = format!("probe failed: {e}");
            None
        }
    };
    let increments = finding
        .gpu_energy_nj_increments_without_root
        .unwrap_or(false);
    EnergyProbeReport {
        root_available: false,
        powermetrics_without_root: "powermetrics must be invoked as the superuser".into(),
        ioreport: finding,
        sample,
        verdict: if increments {
            "NON-ROOT PATH EXISTS: IOReport Energy Model GPU Energy (nJ). It is a GPU rail, not DRAM, and an idle/probe window is not joules_per_token.".into()
        } else {
            "IOReport Energy Model is visible without root but this probe did not see GPU Energy increment. Use sudo powermetrics to fill energy later.".into()
        },
    }
}

#[cfg(not(target_os = "macos"))]
fn read_gpu_energy_nj() -> Result<u64, String> {
    Err("IOReport Energy Model is macOS-only".into())
}

#[cfg(not(target_os = "macos"))]
fn energy_model_channel_count() -> Result<u32, String> {
    Err("IOReport Energy Model is macOS-only".into())
}

#[cfg(target_os = "macos")]
mod ioreport {
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_int, c_void};

    const RTLD_LAZY: c_int = 1;
    const K_CFSTRING_ENCODING_UTF8: u32 = 0x0800_0100;

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFStringCreateWithCString(
            alloc: *const c_void,
            c_str: *const c_char,
            encoding: u32,
        ) -> *mut c_void;
        fn CFDictionaryGetValue(dict: *const c_void, key: *const c_void) -> *const c_void;
        fn CFArrayGetCount(arr: *const c_void) -> isize;
        fn CFArrayGetValueAtIndex(arr: *const c_void, idx: isize) -> *const c_void;
        fn CFGetTypeID(obj: *const c_void) -> usize;
        fn CFDictionaryGetTypeID() -> usize;
        fn CFStringGetCString(
            s: *const c_void,
            buf: *mut c_char,
            buf_size: isize,
            encoding: u32,
        ) -> u8;
        fn CFRelease(obj: *const c_void);
    }

    extern "C" {
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *const c_char;
    }

    type CopyGroup =
        unsafe extern "C" fn(*const c_void, *const c_void, u64, u64, u64) -> *mut c_void;
    type CreateSub = unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        *mut *mut c_void,
        u64,
        *const c_void,
    ) -> *mut c_void;
    type CreateSamples = unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void;
    type GetName = unsafe extern "C" fn(*const c_void) -> *const c_void;
    type SimpleInt = unsafe extern "C" fn(*const c_void, *mut c_int) -> u64;
    type ChannelCount = unsafe extern "C" fn(*const c_void) -> c_int;

    struct Lib {
        copy_group: CopyGroup,
        create_sub: CreateSub,
        create_samples: CreateSamples,
        get_name: GetName,
        simple_int: SimpleInt,
        channel_count: ChannelCount,
    }

    fn load() -> Result<Lib, String> {
        unsafe {
            let path = CString::new("libIOReport.dylib").unwrap();
            let h = dlopen(path.as_ptr(), RTLD_LAZY);
            if h.is_null() {
                let err = dlerror();
                let msg = if err.is_null() {
                    "dlopen libIOReport.dylib failed".into()
                } else {
                    CStr::from_ptr(err).to_string_lossy().into_owned()
                };
                return Err(msg);
            }
            let sym = |name: &str| -> Result<*mut c_void, String> {
                let c = CString::new(name).unwrap();
                let p = dlsym(h, c.as_ptr());
                if p.is_null() {
                    Err(format!("dlsym {name} failed"))
                } else {
                    Ok(p)
                }
            };
            Ok(Lib {
                copy_group: std::mem::transmute(sym("IOReportCopyChannelsInGroup")?),
                create_sub: std::mem::transmute(sym("IOReportCreateSubscription")?),
                create_samples: std::mem::transmute(sym("IOReportCreateSamples")?),
                get_name: std::mem::transmute(sym("IOReportChannelGetChannelName")?),
                simple_int: std::mem::transmute(sym("IOReportSimpleGetIntegerValue")?),
                channel_count: std::mem::transmute(sym("IOReportGetChannelCount")?),
            })
        }
    }

    fn cfstr(s: &str) -> *mut c_void {
        let c = CString::new(s).unwrap();
        unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), K_CFSTRING_ENCODING_UTF8) }
    }

    fn cf_to_string(s: *const c_void) -> Option<String> {
        if s.is_null() {
            return None;
        }
        let mut buf = [0i8; 256];
        let ok = unsafe {
            CFStringGetCString(
                s,
                buf.as_mut_ptr(),
                buf.len() as isize,
                K_CFSTRING_ENCODING_UTF8,
            )
        };
        if ok == 0 {
            return None;
        }
        unsafe { CStr::from_ptr(buf.as_ptr()).to_str().ok().map(|s| s.to_owned()) }
    }

    fn with_samples<T>(f: impl FnOnce(&Lib, *mut c_void) -> Result<T, String>) -> Result<T, String> {
        let lib = load()?;
        unsafe {
            let g = cfstr("Energy Model");
            if g.is_null() {
                return Err("CFString Energy Model failed".into());
            }
            let ch = (lib.copy_group)(g, std::ptr::null(), 0, 0, 0);
            CFRelease(g);
            if ch.is_null() {
                return Err("IOReportCopyChannelsInGroup(Energy Model) returned null".into());
            }
            let mut subbed: *mut c_void = std::ptr::null_mut();
            let sub = (lib.create_sub)(std::ptr::null(), ch, &mut subbed, 0, std::ptr::null());
            if sub.is_null() || subbed.is_null() {
                return Err("IOReportCreateSubscription failed".into());
            }
            let samples = (lib.create_samples)(sub, subbed);
            if samples.is_null() {
                return Err("IOReportCreateSamples returned null".into());
            }
            f(&lib, samples)
        }
    }

    pub fn channel_count() -> Result<u32, String> {
        with_samples(|lib, samples| {
            let n = unsafe { (lib.channel_count)(samples) };
            Ok(n.max(0) as u32)
        })
    }

    pub fn gpu_energy_nj() -> Result<u64, String> {
        with_samples(|lib, samples| unsafe {
            let key = cfstr("IOReportChannels");
            let arr = CFDictionaryGetValue(samples, key);
            CFRelease(key);
            if arr.is_null() {
                return Err("samples missing IOReportChannels".into());
            }
            let n = CFArrayGetCount(arr);
            let dict_tid = CFDictionaryGetTypeID();
            for i in 0..n {
                let item = CFArrayGetValueAtIndex(arr, i);
                if item.is_null() || CFGetTypeID(item) != dict_tid {
                    continue;
                }
                let name = cf_to_string((lib.get_name)(item));
                if name.as_deref() == Some("GPU Energy") {
                    let mut ok: c_int = 0;
                    let v = (lib.simple_int)(item, &mut ok);
                    return Ok(v);
                }
            }
            Err("GPU Energy channel not found".into())
        })
    }
}

#[cfg(target_os = "macos")]
fn read_gpu_energy_nj() -> Result<u64, String> {
    ioreport::gpu_energy_nj()
}

#[cfg(target_os = "macos")]
fn energy_model_channel_count() -> Result<u32, String> {
    ioreport::channel_count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_is_honest() {
        let e = EnergyReport::unavailable("no joules");
        assert!(!e.energy_available);
        assert!(e.joules_per_token.is_none());
        assert!(e.fill_command.contains("sudo powermetrics"));
        assert_eq!(
            e.non_root_ioreport.gpu_energy_nj_increments_without_root,
            Some(true)
        );
    }

    #[test]
    fn caller_joules_mark_available() {
        let e = EnergyReport::from_caller_joules(1.5);
        assert!(e.energy_available);
        assert_eq!(e.joules_per_token, Some(1.5));
        assert_eq!(e.scope, EnergyScope::CallerJoulesPerToken);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ioreport_gpu_energy_is_readable_without_root() {
        let v = read_gpu_energy_nj().expect("GPU Energy readable without root");
        assert!(v > 0, "lifetime GPU Energy should be nonzero on this machine");
    }
}
