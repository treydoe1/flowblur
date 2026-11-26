//! Builds the PiPL resource (target/<profile>/tblur.rsrc), PkgInfo and Info.plist via the
//! `pipl` crate, and exports the PiPL values as env vars that `define_effect!` reports back to
//! the host at PF_Cmd_GLOBAL_SETUP. Because the out-flags and the effect version reported at
//! global setup are read from the same `PIPL_OUTFLAGS`/`PIPL_OUTFLAGS2`/`PIPL_VERSION` values
//! written into the resource, they cannot disagree with the PiPL. The version numbers themselves
//! come from Cargo.toml (`CARGO_PKG_VERSION_*`).
use pipl::*;

// AE effect API spec version the bindings were generated against (PF_PLUG_IN_VERSION /
// PF_PLUG_IN_SUBVERS of the AE SDK the bindings ship), same as the bindings' examples.
const PF_PLUG_IN_VERSION: u16 = 13;
const PF_PLUG_IN_SUBVERS: u16 = 28;

/// One numeric component of the Cargo package version, checked against the width of its field in
/// the PiPL version encoding (pipl masks out-of-range values silently, which would mis-number).
fn pkg_version_field(var: &str, max: u32) -> u32 {
    let raw = std::env::var(var).unwrap_or_else(|_| panic!("{var} not set by cargo"));
    let v: u32 = raw.parse().unwrap_or_else(|_| panic!("{var}={raw} is not a number"));
    assert!(v <= max, "{var}={v} does not fit the PiPL version field (max {max})");
    v
}

#[rustfmt::skip]
fn main() {
    // Keep `catch_unwind` around EffectMain in release builds too (the bindings only enable it
    // in debug builds unless this cfg is set).
    println!("cargo:rustc-cfg=catch_panics");

    // Use `tblur_core::render_with_luma` when ../core/src/lib.rs provides it; otherwise
    // fall back to `render` (see `call_core` in src/lib.rs). Detected automatically.
    println!("cargo::rustc-check-cfg=cfg(core_has_render_with_luma)");
    println!("cargo:rerun-if-changed=../core/src/lib.rs");
    if let Ok(src) = std::fs::read_to_string("../core/src/lib.rs")
        && src.contains("pub fn render_with_luma(") {
        println!("cargo:rustc-cfg=core_has_render_with_luma");
    }
    println!("cargo:rerun-if-changed=build.rs");

    // Give the dylib a neutral install name instead of the absolute path of the build tree.
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/TBlur");

    pipl::plugin_build(vec![
        Property::Kind(PIPLType::AEEffect),
        Property::Name("T Blur"),
        Property::Category("T Blur"),

        #[cfg(target_os = "windows")]
        Property::CodeWin64X86("EffectMain"),
        #[cfg(target_os = "macos")]
        Property::CodeMacARM64("EffectMain"),

        Property::AE_PiPL_Version { major: 2, minor: 0 },
        Property::AE_Effect_Spec_Version { major: PF_PLUG_IN_VERSION, minor: PF_PLUG_IN_SUBVERS },
        // From `version` in Cargo.toml. pipl encodes these same fields both into the resource and
        // into PIPL_VERSION, which define_effect! reports as out_data.my_version at global setup.
        Property::AE_Effect_Version {
            version: pkg_version_field("CARGO_PKG_VERSION_MAJOR", 127),
            subversion: pkg_version_field("CARGO_PKG_VERSION_MINOR", 15),
            bugversion: pkg_version_field("CARGO_PKG_VERSION_PATCH", 15),
            // The project is an alpha.
            stage: Stage::Alpha,
            build: 1,
        },
        Property::AE_Effect_Info_Flags(0),
        Property::AE_Effect_Global_OutFlags(
            // We check out the input layer at other times.
            OutFlags::WideTimeInput |
            // We handle 16-bpc worlds in After Effects (Premiere ignores this, see lib.rs).
            OutFlags::DeepColorAware
        ),
        Property::AE_Effect_Global_OutFlags_2(
            // Safe for AE Multi-Frame Rendering: the only shared state (the Premiere-only per-clip cache
            // of source-frame verdicts) is mutex-guarded.
            OutFlags2::SupportsThreadedRendering |
            // Required alongside threaded rendering (the pipl crate warns without it).
            OutFlags2::SupportsGetFlattenedSequenceData
        ),
        // Must never change: projects reference the effect by this name.
        Property::AE_Effect_Match_Name("TBLUR Motion Blur"),
        Property::AE_Reserved_Info(8),
        Property::AE_Effect_Support_URL("https://github.com/treydoe1/tblur"),
    ])
}
