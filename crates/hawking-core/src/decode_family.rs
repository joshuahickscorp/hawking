//! G023 shared decode family: one Rust path for Qwen (Q80 + Qwen3.8) and DSV4F.
//!
//! Metal source lives in `shaders/gk_family.metal`. Per-model differences are
//! specialization constants / named specializations, not forked inner loops.
//! STRUCTURAL leftovers (rice CSR, two-stage hgravs, MHC, act-quant, Q4)
//! stay on their original kernels.

/// Schema for receipts that claim a graph ran on the shared family.
pub const SCHEMA: &str = "hawking.decode_family.v1";

/// Q80 / Qwen3.8 binary_group serial matvec (shipping mixed graph).
pub const MATVEC_BINARY: &str = "gk_matvec_binary";
/// Family 1-SG/row tile of [`MATVEC_BINARY`]. Opt-in via
/// `HAWKING_Q80_GK_SIMD=1`; 2.5× slower than [`MATVEC_BINARY_TILES`].
pub const MATVEC_BINARY_SIMD: &str = "gk_matvec_binary_simd";
/// Occupancy tile of [`MATVEC_BINARY`]. Shipping Q80 recon-fuse default
/// (256 threads/row, in-register consume). Lives in `q80_mixed_decode.metal`.
pub const MATVEC_BINARY_TILES: &str = "q80_binary_group_matvec_tg256";
/// Occupancy tile of fused binary + rice CSR residual.
pub const MATVEC_BINARY_CSR_TILES: &str = "q80_binary_group_csr_matvec_tg256";
/// Q80 / Qwen3.8 hgravs01 / uniform-n factor serial matvec.
pub const MATVEC_HGRAVS: &str = "gk_matvec_hgravs";
pub const MATVEC_HGRAVS_SIMD: &str = "gk_matvec_hgravs_simd";
/// Occupancy tile of 3-bit hgravs / uniform factor (8-unpack).
pub const MATVEC_HGRAVS_TILES: &str = "q80_hgravs01_factor_matvec_simd3";
/// Occupancy tile of n-bit hgravs / uniform factor (not 3-bit, not wide Q8).
pub const MATVEC_HGRAVS_NBIT_TILES: &str = "q80_hgravs01_factor_matvec_simd";
/// Occupancy tile of uniform-q8 for wide columns (`cols >= 2048`).
pub const MATVEC_UNIFORM8_TILES: &str = "q80_uniform8_matvec_tg256";
/// Occupancy tile of uniform-q8 for narrow columns (byte extract).
pub const MATVEC_UNIFORM8_SIMD_BYTES: &str = "q80_uniform8_matvec_simd_bytes";
/// Isolated DSV4F FP4 row (same association as one worklist slot).
pub const MATVEC_FP4: &str = "gk_matvec_fp4";
/// DSV4F compact-K worklist FP4 (default K=6).
pub const WORKLIST_FP4: &str = "gk_worklist_fp4";
pub const WORKLIST_FP4_SIMD: &str = "gk_worklist_fp4_simd";
/// Q80 / Qwen3.8 / dense SwiGLU.
pub const SWIGLU_F32: &str = "gk_swiglu_f32";
/// DSV4F worklist SwiGLU (bf16, clamp, route weight).
pub const SWIGLU_BF16_WORKLIST: &str = "gk_swiglu_bf16_worklist";
/// DSV4F (and Q80-if-worklisted) combine.
pub const COMBINE_BF16: &str = "gk_combine_bf16";
/// Device worklist pack. Default K=6; specialize kGkWorklistK=10 for Q80.
pub const PACK_WORKLIST: &str = "gk_pack_worklist";

/// Kernels the Q80 mixed hybrid graph must dispatch after G023.
/// Serial names are the `HAWKING_Q80_RECON_FUSE=0` fallback. Occupancy
/// tiles ([`Q80_TILE_KERNELS`]) are the default. `*_SIMD` is the unused
/// family 1-SG/row tile, opt-in via `HAWKING_Q80_GK_SIMD=1`.
pub const Q80_GRAPH_KERNELS: &[&str] = &[MATVEC_BINARY, MATVEC_HGRAVS];
pub const Q80_GRAPH_SIMD_KERNELS: &[&str] = &[MATVEC_BINARY_SIMD, MATVEC_HGRAVS_SIMD];
/// Shipping Q80 recon-fuse occupancy tiles. Same codecs as
/// [`Q80_GRAPH_KERNELS`]; better launch geometry. Not in
/// [`FAMILY_KERNELS`] — the Metal lives in `q80_mixed_decode.metal`.
pub const Q80_TILE_KERNELS: &[&str] = &[
    MATVEC_BINARY_TILES,
    MATVEC_BINARY_CSR_TILES,
    MATVEC_HGRAVS_TILES,
    MATVEC_HGRAVS_NBIT_TILES,
    MATVEC_UNIFORM8_TILES,
    MATVEC_UNIFORM8_SIMD_BYTES,
];

/// Dense Qwen3.8 mixed graph. Same family matvecs as Q80; no worklist,
/// pack, or expert combine. Occupancy tiles stay in `q80_mixed_decode.metal`.
pub const QWEN38_GRAPH_KERNELS: &[&str] = &[MATVEC_BINARY, MATVEC_HGRAVS];

/// Kernels the DSV4F native token graph must dispatch after G023.
pub const DSV4F_GRAPH_KERNELS: &[&str] = &[
    PACK_WORKLIST,
    WORKLIST_FP4,
    WORKLIST_FP4_SIMD,
    SWIGLU_BF16_WORKLIST,
    COMBINE_BF16,
];

/// Every family entry point. Used by the compile/trace-name contract.
pub const FAMILY_KERNELS: &[&str] = &[
    MATVEC_BINARY,
    MATVEC_BINARY_SIMD,
    MATVEC_HGRAVS,
    MATVEC_HGRAVS_SIMD,
    MATVEC_FP4,
    WORKLIST_FP4,
    WORKLIST_FP4_SIMD,
    SWIGLU_F32,
    SWIGLU_BF16_WORKLIST,
    COMBINE_BF16,
    PACK_WORKLIST,
];

/// True if `name` is a G023 family kernel (not a pre-family alias).
pub fn is_family_kernel(name: &str) -> bool {
    FAMILY_KERNELS.contains(&name)
}

/// Default **on**. Dispatch the G023 family entry points. Set
/// `HAWKING_DECODE_FAMILY=0` / `false` / `off` / `no` to dispatch the
/// pre-family wrapper names (same helpers, old kernel symbols).
///
/// This is a name-only switch. Host bind, recon-fuse occupancy tiles,
/// command-buffer fusion, and act-quant stay on their own levers.
pub fn family_dispatch_enabled() -> bool {
    crate::env_opt_out("HAWKING_DECODE_FAMILY")
}

pub const LEGACY_MATVEC_BINARY: &str = "q80_binary_group_matvec";
pub const LEGACY_MATVEC_HGRAVS: &str = "q80_hgravs01_factor_matvec";
pub const LEGACY_PACK_WORKLIST: &str = "dsv4f_pack_worklist";
pub const LEGACY_WORKLIST_FP4: &str = "dsv4f_worklist_fp4_matvec";
pub const LEGACY_WORKLIST_FP4_SIMD: &str = "dsv4f_worklist_fp4_matvec_simd";
pub const LEGACY_SWIGLU_F32: &str = "qwen80_silu_mul_f32";
pub const LEGACY_SWIGLU_BF16_WORKLIST: &str = "dsv4f_worklist_swiglu";
pub const LEGACY_COMBINE_BF16: &str = "dsv4f_worklist_combine";

fn pick<'a>(family: &'a str, legacy: &'a str) -> &'a str {
    if family_dispatch_enabled() {
        family
    } else {
        legacy
    }
}

pub fn matvec_binary() -> &'static str {
    pick(MATVEC_BINARY, LEGACY_MATVEC_BINARY)
}

pub fn matvec_binary_tiles() -> &'static str {
    MATVEC_BINARY_TILES
}

pub fn matvec_binary_csr_tiles() -> &'static str {
    MATVEC_BINARY_CSR_TILES
}

pub fn matvec_hgravs() -> &'static str {
    pick(MATVEC_HGRAVS, LEGACY_MATVEC_HGRAVS)
}

pub fn matvec_hgravs_tiles() -> &'static str {
    MATVEC_HGRAVS_TILES
}

pub fn matvec_hgravs_nbit_tiles() -> &'static str {
    MATVEC_HGRAVS_NBIT_TILES
}

pub fn matvec_uniform8_tiles() -> &'static str {
    MATVEC_UNIFORM8_TILES
}

pub fn matvec_uniform8_simd_bytes() -> &'static str {
    MATVEC_UNIFORM8_SIMD_BYTES
}

pub fn pack_worklist() -> &'static str {
    pick(PACK_WORKLIST, LEGACY_PACK_WORKLIST)
}

pub fn worklist_fp4() -> &'static str {
    pick(WORKLIST_FP4, LEGACY_WORKLIST_FP4)
}

pub fn worklist_fp4_simd() -> &'static str {
    pick(WORKLIST_FP4_SIMD, LEGACY_WORKLIST_FP4_SIMD)
}

pub fn swiglu_f32() -> &'static str {
    pick(SWIGLU_F32, LEGACY_SWIGLU_F32)
}

pub fn swiglu_bf16_worklist() -> &'static str {
    pick(SWIGLU_BF16_WORKLIST, LEGACY_SWIGLU_BF16_WORKLIST)
}

pub fn combine_bf16() -> &'static str {
    pick(COMBINE_BF16, LEGACY_COMBINE_BF16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_kernels_are_in_the_shader() {
        let src = crate::metal::SHADER_GK_FAMILY;
        for &kernel in FAMILY_KERNELS {
            assert!(
                src.contains(&format!("kernel void {kernel}(")),
                "{kernel} missing from gk_family.metal"
            );
        }
    }

    #[test]
    fn both_graphs_name_only_family_kernels() {
        for &kernel in Q80_GRAPH_KERNELS
            .iter()
            .chain(QWEN38_GRAPH_KERNELS)
            .chain(DSV4F_GRAPH_KERNELS)
        {
            assert!(is_family_kernel(kernel), "{kernel} is not a family kernel");
        }
        assert_eq!(QWEN38_GRAPH_KERNELS, Q80_GRAPH_KERNELS);
    }

    #[test]
    fn occupancy_tiles_keep_the_shipping_q80_names() {
        assert_eq!(MATVEC_BINARY_TILES, "q80_binary_group_matvec_tg256");
        assert_eq!(
            MATVEC_BINARY_CSR_TILES,
            "q80_binary_group_csr_matvec_tg256"
        );
        assert_eq!(MATVEC_HGRAVS_TILES, "q80_hgravs01_factor_matvec_simd3");
        assert_eq!(
            MATVEC_HGRAVS_NBIT_TILES,
            "q80_hgravs01_factor_matvec_simd"
        );
        assert_eq!(MATVEC_UNIFORM8_TILES, "q80_uniform8_matvec_tg256");
        assert_eq!(
            MATVEC_UNIFORM8_SIMD_BYTES,
            "q80_uniform8_matvec_simd_bytes"
        );
        assert_eq!(matvec_binary_tiles(), MATVEC_BINARY_TILES);
        assert_eq!(matvec_binary_csr_tiles(), MATVEC_BINARY_CSR_TILES);
        assert_eq!(matvec_hgravs_tiles(), MATVEC_HGRAVS_TILES);
        assert_eq!(matvec_hgravs_nbit_tiles(), MATVEC_HGRAVS_NBIT_TILES);
        assert_eq!(matvec_uniform8_tiles(), MATVEC_UNIFORM8_TILES);
        assert_eq!(matvec_uniform8_simd_bytes(), MATVEC_UNIFORM8_SIMD_BYTES);
        let src = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/q80_mixed_decode.metal"
        ));
        for &kernel in Q80_TILE_KERNELS {
            assert!(
                src.contains(&format!("kernel void {kernel}(")),
                "{kernel} missing from q80_mixed_decode.metal"
            );
        }
    }

    #[test]
    fn legacy_aliases_are_the_pre_family_entry_points() {
        assert_eq!(LEGACY_MATVEC_BINARY, "q80_binary_group_matvec");
        assert_eq!(LEGACY_MATVEC_HGRAVS, "q80_hgravs01_factor_matvec");
        assert_eq!(LEGACY_PACK_WORKLIST, "dsv4f_pack_worklist");
        assert_eq!(LEGACY_WORKLIST_FP4, "dsv4f_worklist_fp4_matvec");
        assert_eq!(LEGACY_WORKLIST_FP4_SIMD, "dsv4f_worklist_fp4_matvec_simd");
        assert_eq!(LEGACY_SWIGLU_F32, "qwen80_silu_mul_f32");
        assert_eq!(LEGACY_SWIGLU_BF16_WORKLIST, "dsv4f_worklist_swiglu");
        assert_eq!(LEGACY_COMBINE_BF16, "dsv4f_worklist_combine");
    }

    #[test]
    fn qwen38_geometry_is_an_admitted_specialization() {
        let src = crate::metal::SHADER_GK_FAMILY;
        assert!(src.contains("n_heads == 24u && n_kv_heads == 4u"));
        assert!(src.contains("rope_theta == 10000000.0f"));
        let gates = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/kernels/mod.rs"
        ));
        assert!(gates.contains("matches!(heads, 32 | 48)"));
        assert!(gates.contains("matches!(values_per_key_head, 2 | 3)"));
    }
}
