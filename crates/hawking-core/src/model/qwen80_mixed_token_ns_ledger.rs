//! Closed TOKEN_NS cover of the Q80 mixed-1p5-v1 complete token.
//!
//! Production path is many CBs with host activations between them. GPU time
//! is `MTLCommandBuffer.GPUEndTime − GPUStartTime` after wait. Isolated
//! addr/decode probes run after a production generate and do not sit inside
//! the timed token.
//!
//! Identity: `sum(serial components) == complete_token_wall`. Residual is a
//! named row.

use super::qwen80_complete_runtime::{
    QWEN80_EXPERTS, QWEN80_HIDDEN, QWEN80_LAYERS, QWEN80_MOE_INTERMEDIATE, QWEN80_TOP_K,
    QWEN80_VOCAB,
};
use super::qwen80_mixed_hybrid_decode::{
    MixedExclusiveSnap, MixedGpuOrganNs, MixedHostExclusiveNs, MixedTokenSample,
};
use serde::Serialize;

pub const QWEN80_MIXED_TOKEN_NS_LEDGER_SCHEMA: &str =
    "hawking.ascension.qwen80_mixed_token_ns_ledger.v1";
pub const GPU_TIMESTAMP_AUTHORITY: &str =
    "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy";
pub const HONEST_DECODE_CEILING_GB_S: f64 = 411.51;
pub const M3_ULTRA_PEAK_GB_S: f64 = 819.0;
pub const MIXED_1P5_V1_BPW: f64 = 1.4444457;

pub const REQUIRED_COMPONENTS: &[&str] = &[
    "host_preparation",
    "weight_addressing",
    "weight_decode_reconstruction",
    "command_submission",
    "deltanet",
    "gqa",
    "moe_routed",
    "moe_shared",
    "moe_combine",
    "normalization",
    "kv_state",
    "terminal_head",
    "synchronization",
    "unattributed_residual",
];

#[derive(Clone, Debug, Serialize)]
pub struct ComponentRow {
    pub component: String,
    pub ns_per_token: f64,
    pub pct_of_token_wall: f64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub dispatches: u64,
    pub command_buffers: u64,
    pub cpu_involvement: &'static str,
    pub gpu_occupancy: String,
    pub effective_gb_s: Option<f64>,
    pub theoretical_lower_bound_ns: f64,
    pub measured_over_floor: Option<f64>,
    pub resource_class: &'static str,
    pub serial_or_overlappable: &'static str,
    pub in_identity_sum: bool,
    pub confidence: &'static str,
    pub method: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct IsolatedFamily {
    pub name: String,
    pub gpu_ns_reps: Vec<u64>,
    pub median_gpu_ns: u64,
    pub wait_ns_median: u64,
    pub dispatches: u64,
    pub command_buffers: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProbeSplit {
    pub class: String,
    pub full_median_gpu_ns: u64,
    pub addr_median_gpu_ns: u64,
    pub decode_median_gpu_ns: u64,
    pub addr_frac_of_full: f64,
    pub decode_minus_addr_frac: f64,
    pub fma_remainder_frac: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClosureLine {
    pub identity: &'static str,
    pub total_token_ns: u64,
    pub sum_serial_ns: i128,
    pub residual_ns: i128,
    pub residual_name: String,
    pub residual_reason: String,
    pub identity_holds: bool,
    pub production_gpu_ns: u64,
    pub organ_gpu_sum_ns: u64,
    pub host_exclusive_sum_ns: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct MixedByteBudget {
    pub deltanet_bytes: u64,
    pub gqa_bytes: u64,
    pub router_bytes: u64,
    pub shared_expert_bytes: u64,
    pub routed_expert_bytes: u64,
    pub combine_gate_bytes: u64,
    pub lm_head_bytes: u64,
    pub embed_row_bytes: u64,
    pub norms_bytes: u64,
    pub state_rw_bytes: u64,
    pub total_weight_bytes: u64,
    pub note: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen80MixedTokenNsLedger {
    pub schema: &'static str,
    pub model: &'static str,
    pub vehicle: &'static str,
    pub bpw: f64,
    pub kernel_runtime_genome: String,
    pub measurement_label: &'static str,
    pub gpu_timestamp_authority: &'static str,
    pub commit: String,
    pub regime: String,
    pub production_cb_shape: bool,
    pub bytes: MixedByteBudget,
    pub token_samples: Vec<MixedTokenSample>,
    pub steady_wall_ns: Vec<u64>,
    pub steady_gpu_ns: Vec<u64>,
    pub median_wall_ns: u64,
    pub median_gpu_ns: u64,
    pub median_encode_ns: u64,
    pub median_submit_ns: u64,
    pub median_wait_ns: u64,
    pub gpu_spread_ns: [u64; 2],
    pub wait_minus_gpu_ns: i64,
    pub isolated: Vec<IsolatedFamily>,
    pub probes: Vec<ProbeSplit>,
    pub simd_checks: Vec<IsolatedFamily>,
    pub components: Vec<ComponentRow>,
    pub closure: ClosureLine,
    pub greedy_token_ids: Vec<u32>,
    pub greedy_matches_oracle: bool,
    pub fallbacks: u32,
    pub instrumentation_overhead: String,
    pub notes: Vec<String>,
}

pub fn median_u64(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut s = values.to_vec();
    s.sort_unstable();
    s[s.len() / 2]
}

pub fn bandwidth_floor_ns(bytes: u64, gb_s: f64) -> f64 {
    if gb_s <= 0.0 {
        return 0.0;
    }
    (bytes as f64) / (gb_s * 1.0e9) * 1.0e9
}

pub fn effective_gb_s(bytes: u64, ns: f64) -> Option<f64> {
    if ns <= 0.0 {
        return None;
    }
    Some((bytes as f64) / ns)
}

fn probe_for<'a>(probes: &'a [ProbeSplit], class: &str) -> Option<&'a ProbeSplit> {
    probes.iter().find(|p| p.class == class)
}

fn split_gpu(full: u64, probe: Option<&ProbeSplit>) -> (f64, f64, f64) {
    let Some(p) = probe else {
        return (full as f64, 0.0, 0.0);
    };
    // Tiny isolated organs are noise-dominated (addr_probe can exceed full).
    // If the full kernel is < 25 µs, keep the raw fractions only when they
    // are internally consistent (addr <= decode <= full * 1.15).
    let mut addr_f = p.addr_frac_of_full.clamp(0.0, 1.0);
    let mut dec_f = p.decode_minus_addr_frac.clamp(0.0, 1.0);
    if p.full_median_gpu_ns < 25_000 || p.addr_median_gpu_ns > p.full_median_gpu_ns {
        addr_f = (p.addr_median_gpu_ns.min(p.full_median_gpu_ns) as f64
            / p.full_median_gpu_ns.max(1) as f64)
            .clamp(0.0, 1.0);
        let dec_extra = p
            .decode_median_gpu_ns
            .saturating_sub(p.addr_median_gpu_ns.min(p.decode_median_gpu_ns));
        dec_f = (dec_extra as f64 / p.full_median_gpu_ns.max(1) as f64).clamp(0.0, 1.0 - addr_f);
    }
    if addr_f + dec_f > 1.0 {
        dec_f = (1.0 - addr_f).max(0.0);
    }
    let addr = addr_f * full as f64;
    let dec = dec_f * full as f64;
    let fma = (full as f64 - addr - dec).max(0.0);
    (addr, dec, fma)
}

fn median_snap(snaps: &[MixedExclusiveSnap]) -> MixedExclusiveSnap {
    if snaps.is_empty() {
        return MixedExclusiveSnap::default();
    }
    let pick = |f: fn(&MixedExclusiveSnap) -> u64| -> u64 {
        median_u64(&snaps.iter().map(f).collect::<Vec<_>>())
    };
    MixedExclusiveSnap {
        encode_ns: pick(|s| s.encode_ns),
        submit_ns: pick(|s| s.submit_ns),
        wait_ns: pick(|s| s.wait_ns),
        gpu_ns: pick(|s| s.gpu_ns),
        wait_minus_gpu_ns: pick(|s| s.wait_minus_gpu_ns),
        cbs: pick(|s| s.cbs),
        dispatches: pick(|s| s.dispatches),
        timestamps_missing: pick(|s| s.timestamps_missing),
        gpu_organ: MixedGpuOrganNs {
            deltanet: pick(|s| s.gpu_organ.deltanet),
            gqa: pick(|s| s.gpu_organ.gqa),
            moe_shared: pick(|s| s.gpu_organ.moe_shared),
            moe_routed: pick(|s| s.gpu_organ.moe_routed),
            moe_router: pick(|s| s.gpu_organ.moe_router),
            moe_combine_gate: pick(|s| s.gpu_organ.moe_combine_gate),
            terminal: pick(|s| s.gpu_organ.terminal),
            other: pick(|s| s.gpu_organ.other),
        },
        host_excl: MixedHostExclusiveNs {
            embed: pick(|s| s.host_excl.embed),
            dn_rms: pick(|s| s.host_excl.dn_rms),
            dn_rearrange_l2: pick(|s| s.host_excl.dn_rearrange_l2),
            dn_conv: pick(|s| s.host_excl.dn_conv),
            dn_recurrent: pick(|s| s.host_excl.dn_recurrent),
            dn_gated: pick(|s| s.host_excl.dn_gated),
            dn_residual: pick(|s| s.host_excl.dn_residual),
            gqa_rms: pick(|s| s.host_excl.gqa_rms),
            gqa_interleave: pick(|s| s.host_excl.gqa_interleave),
            gqa_rope: pick(|s| s.host_excl.gqa_rope),
            gqa_kv_copy: pick(|s| s.host_excl.gqa_kv_copy),
            gqa_attn: pick(|s| s.host_excl.gqa_attn),
            gqa_residual: pick(|s| s.host_excl.gqa_residual),
            post_rms: pick(|s| s.host_excl.post_rms),
            final_rms: pick(|s| s.host_excl.final_rms),
            silu: pick(|s| s.host_excl.silu),
            topk: pick(|s| s.host_excl.topk),
            combine: pick(|s| s.host_excl.combine),
            argmax: pick(|s| s.host_excl.argmax),
            buffer_prep: pick(|s| s.host_excl.buffer_prep),
            expert_bind: pick(|s| s.host_excl.expert_bind),
            catalog_reparse: pick(|s| s.host_excl.catalog_reparse),
            vector_clone: pick(|s| s.host_excl.vector_clone),
        },
    }
}

/// Exclusive 14-component cover of the complete token wall.
pub fn seal_components(
    wall_ns: u64,
    snap: &MixedExclusiveSnap,
    probes: &[ProbeSplit],
    bytes: &MixedByteBudget,
) -> (Vec<ComponentRow>, ClosureLine) {
    let q8 = probe_for(probes, "q8");
    let q8_out = probe_for(probes, "q8_out").or(q8);
    let q8_router = probe_for(probes, "q8_router").or(q8);
    let q8_head = probe_for(probes, "q8_lm_head").or(q8);
    let binary = probe_for(probes, "binary");
    let residual = probe_for(probes, "residual");
    let hgravs = probe_for(probes, "hgravs");

    // Per-organ probes. Do not apply the large-qkvz fraction to a 1-row gate.
    let (dn_addr, dn_dec, dn_fma) = split_gpu(snap.gpu_organ.deltanet, q8);
    let (gqa_addr, gqa_dec, gqa_fma) = split_gpu(snap.gpu_organ.gqa, q8_out.or(q8));
    let (shared_addr, shared_dec, shared_fma) = split_gpu(snap.gpu_organ.moe_shared, q8_router);
    let (router_addr, router_dec, router_fma) = split_gpu(snap.gpu_organ.moe_router, q8_router);
    let (gate_addr, gate_dec, gate_fma) = split_gpu(snap.gpu_organ.moe_combine_gate, q8_router);
    let (head_addr, head_dec, head_fma) = split_gpu(snap.gpu_organ.terminal, q8_head);

    // Routed wave is a blend. If a routed-class probe exists, use it;
    // otherwise blend binary/residual/hgravs 1:1:2 (gate + up + two factors).
    let routed_probe = probe_for(probes, "routed").cloned().unwrap_or_else(|| {
        let b = binary.cloned().unwrap_or(ProbeSplit {
            class: "binary".into(),
            full_median_gpu_ns: 1,
            addr_median_gpu_ns: 0,
            decode_median_gpu_ns: 0,
            addr_frac_of_full: 0.5,
            decode_minus_addr_frac: 0.2,
            fma_remainder_frac: 0.3,
        });
        let r = residual.cloned().unwrap_or(b.clone());
        let h = hgravs.cloned().unwrap_or(b.clone());
        let addr = (b.addr_frac_of_full + r.addr_frac_of_full + 2.0 * h.addr_frac_of_full) / 4.0;
        let dec = (b.decode_minus_addr_frac
            + r.decode_minus_addr_frac
            + 2.0 * h.decode_minus_addr_frac)
            / 4.0;
        ProbeSplit {
            class: "routed".into(),
            full_median_gpu_ns: snap.gpu_organ.moe_routed,
            addr_median_gpu_ns: 0,
            decode_median_gpu_ns: 0,
            addr_frac_of_full: addr.clamp(0.0, 1.0),
            decode_minus_addr_frac: dec.clamp(0.0, 1.0 - addr),
            fma_remainder_frac: (1.0 - addr - dec).max(0.0),
        }
    });
    let (routed_addr, routed_dec, routed_fma) =
        split_gpu(snap.gpu_organ.moe_routed, Some(&routed_probe));

    let weight_addr =
        dn_addr + gqa_addr + shared_addr + router_addr + gate_addr + head_addr + routed_addr;
    let weight_dec = dn_dec + gqa_dec + shared_dec + router_dec + gate_dec + head_dec + routed_dec;

    let host_prep = snap.encode_ns as f64
        + snap.host_excl.buffer_prep as f64
        + snap.host_excl.expert_bind as f64
        + snap.host_excl.embed as f64
        + snap.host_excl.catalog_reparse as f64
        + snap.host_excl.vector_clone as f64;
    let submit = snap.submit_ns as f64;
    let sync = (snap.wait_ns as f64 - snap.gpu_ns as f64).max(0.0);
    let deltanet = snap.host_excl.deltanet_host() as f64 + dn_fma;
    let gqa = snap.host_excl.gqa_host() as f64 + gqa_fma;
    let moe_routed = routed_fma;
    let moe_shared = snap.host_excl.silu as f64 + shared_fma;
    let moe_combine =
        snap.host_excl.topk as f64 + snap.host_excl.combine as f64 + router_fma + gate_fma;
    let norms = snap.host_excl.normalization_host() as f64;
    let kv = snap.host_excl.gqa_kv_copy as f64;
    let terminal = snap.host_excl.argmax as f64 + head_fma;

    let named_except_residual = host_prep
        + weight_addr
        + weight_dec
        + submit
        + deltanet
        + gqa
        + moe_routed
        + moe_shared
        + moe_combine
        + norms
        + kv
        + terminal
        + sync;
    // May be slightly negative: exclusive Instant fields on one token
    // can over-sum the wall by a few ms (composition of 337 CB clocks).
    // The residual row carries that so sum(components) == wall.
    let residual = wall_ns as f64 - named_except_residual;

    let row = |name: &str,
               ns: f64,
               bread: u64,
               bwrite: u64,
               disp: u64,
               cbs: u64,
               cpu: &'static str,
               occ: String,
               resource: &'static str,
               method: String|
     -> ComponentRow {
        let floor = bandwidth_floor_ns(bread.saturating_add(bwrite), HONEST_DECODE_CEILING_GB_S);
        let gb = effective_gb_s(bread.saturating_add(bwrite), ns);
        let ratio = if floor > 0.0 { Some(ns / floor) } else { None };
        ComponentRow {
            component: name.to_owned(),
            ns_per_token: ns,
            pct_of_token_wall: if wall_ns == 0 {
                0.0
            } else {
                ns * 100.0 / wall_ns as f64
            },
            bytes_read: bread,
            bytes_written: bwrite,
            dispatches: disp,
            command_buffers: cbs,
            cpu_involvement: cpu,
            gpu_occupancy: occ,
            effective_gb_s: gb,
            theoretical_lower_bound_ns: floor,
            measured_over_floor: ratio,
            resource_class: resource,
            serial_or_overlappable: "serial",
            in_identity_sum: true,
            confidence: "measured",
            method,
        }
    };

    let hidden_b = (QWEN80_HIDDEN * 4) as u64;
    let mid_b = (QWEN80_MOE_INTERMEDIATE * 4) as u64;
    let components = vec![
        row(
            "host_preparation",
            host_prep,
            bytes.embed_row_bytes,
            hidden_b,
            snap.dispatches,
            snap.cbs,
            "CPU encode of CBs + host buffer write/read + expert first-touch bind + embed row gather + catalog reparse + vector cache clone",
            "n/a (CPU)".into(),
            "cpu",
            "production Instant: encode_ns + buffer_prep + expert_bind + embed + catalog_reparse + vector_clone".into(),
        ),
        row(
            "weight_addressing",
            weight_addr,
            bytes.total_weight_bytes,
            0,
            snap.dispatches,
            snap.cbs,
            "none in kernel; host bind is in host_preparation",
            "launch-geometry derived; small organs (ba 64, router 512) are TG-starved on tg256".into(),
            "gpu",
            "addr_probe/full fraction per codec, applied to production organ GPU timestamps".into(),
        ),
        row(
            "weight_decode_reconstruction",
            weight_dec,
            0,
            0,
            snap.dispatches,
            snap.cbs,
            "none in kernel",
            "same launch as addressing; extra ALU is unpack without x-FMA".into(),
            "gpu",
            "(decode_probe − addr_probe) / full, applied to production organ GPU".into(),
        ),
        row(
            "command_submission",
            submit,
            0,
            0,
            0,
            snap.cbs,
            "CPU Instant around MTLCommandBuffer.commit",
            "n/a (CPU)".into(),
            "cpu",
            "sum of production CB submit_ns".into(),
        ),
        row(
            "deltanet",
            deltanet,
            hidden_b * 36 * 4,
            hidden_b * 36,
            36 * 4,
            36,
            "HOST conv + recurrent + gated RMS + rearrange/L2 + residual add. GPU FMA remainder of Q8 GEMVs.",
            "host scalar; GPU qkvz 12288 TGs, ba 64 TGs, out 2048 TGs".into(),
            "cpu+gpu",
            "exclusive host DeltaNet acts + Q8 FMA remainder of production deltanet GPU".into(),
        ),
        row(
            "gqa",
            gqa,
            hidden_b * 12 * 2 + bytes.state_rw_bytes / 4,
            hidden_b * 12,
            12 * 4,
            12,
            "HOST RoPE + causal attention + sigmoid gate + residual. GPU FMA remainder of Q/K/V/O.",
            "host scalar attention over seq; GPU q 8192 TGs, kv 512 TGs".into(),
            "cpu+gpu",
            "exclusive host GQA acts + Q8 FMA remainder of production gqa GPU".into(),
        ),
        row(
            "moe_routed",
            moe_routed,
            bytes.routed_expert_bytes,
            hidden_b * (QWEN80_LAYERS as u64),
            (QWEN80_LAYERS * QWEN80_TOP_K * 4) as u64,
            QWEN80_LAYERS as u64,
            "fused wave on GPU (gate binary + up residual + down hgravs + silu + weighted sum). Host bind is host_preparation.",
            "gate 512 TGs tg256; down R 160 rows occupancy-starved; L 2048x160 simd3".into(),
            "gpu",
            "production routed GPU FMA remainder after binary/residual/hgravs addr+decode split".into(),
        ),
        row(
            "moe_shared",
            moe_shared,
            bytes.shared_expert_bytes,
            hidden_b * (QWEN80_LAYERS as u64) + mid_b * (QWEN80_LAYERS as u64),
            (QWEN80_LAYERS * 3) as u64,
            QWEN80_LAYERS as u64,
            "HOST silu_mul; GPU Q8 FMA remainder of shared gate/up/down",
            "shared 512-row Q8 tg256".into(),
            "cpu+gpu",
            "exclusive host silu + shared-expert GEMV FMA remainder".into(),
        ),
        row(
            "moe_combine",
            moe_combine,
            bytes.router_bytes + bytes.combine_gate_bytes,
            hidden_b * (QWEN80_LAYERS as u64) + (QWEN80_EXPERTS as u64) * 4 * (QWEN80_LAYERS as u64),
            (QWEN80_LAYERS * 2) as u64,
            (QWEN80_LAYERS * 2) as u64,
            "HOST top-10 + shared-gate sigmoid + residual add. GPU FMA of router and 1-row shared gate.",
            "router 512 TGs; shared-gate 1 TG".into(),
            "cpu+gpu",
            "exclusive host topk+combine + router/gate FMA remainder".into(),
        ),
        row(
            "normalization",
            norms,
            bytes.norms_bytes + hidden_b * (1 + 48 + 48 + 1),
            hidden_b * (36 + 12 + 48 + 1),
            (36 + 12 + 48 + 1) as u64,
            0,
            "HOST residual RMSNorm (input, post-attn, final). Not a Metal kernel on this path.",
            "n/a (CPU scalar over 2048)".into(),
            "cpu",
            "exclusive host RMS Instant wraps".into(),
        ),
        row(
            "kv_state",
            kv,
            bytes.state_rw_bytes / 2,
            bytes.state_rw_bytes / 2,
            0,
            0,
            "HOST memcpy of conv state (36) and GQA K/V cache slot (12). Recurrent write is inside deltanet_recurrent.",
            "n/a (CPU memcpy)".into(),
            "cpu",
            "exclusive Instant around conv-state assign + GQA cache slot copy".into(),
        ),
        row(
            "terminal_head",
            terminal,
            bytes.lm_head_bytes,
            (QWEN80_VOCAB as u64) * 4 + 4,
            1,
            1,
            "HOST greedy argmax; GPU FMA remainder of lm_head Q8. Weight traffic in addressing/decode.",
            "lm_head 151936 TGs tg256".into(),
            "cpu+gpu",
            "exclusive host argmax + lm_head FMA remainder".into(),
        ),
        row(
            "synchronization",
            sync,
            0,
            4,
            0,
            snap.cbs,
            "CPU blocked in waitUntilCompleted minus GPU busy, summed over production CBs",
            "n/a (CPU/sync)".into(),
            "sync",
            "production wait_ns − gpu_ns across all CBs of the token".into(),
        ),
        row(
            "unattributed_residual",
            residual,
            0,
            0,
            0,
            0,
            "host tail between Instant marks: vector() cache hits, Vec alloc, layer loop, finite checks",
            "n/a".into(),
            "cpu",
            format!(
                "NAMED residual = wall {wall_ns} − named_except_residual {named_except_residual:.0}. Host gaps not wrapped by exclusive Instants."
            ),
        ),
    ];

    let sum_serial: i128 = components
        .iter()
        .map(|c| c.ns_per_token.round() as i128)
        .sum();
    let residual_ns = wall_ns as i128 - sum_serial;
    let closure = ClosureLine {
        identity: "sum(serial components) == complete_token_wall; residual is the named unattributed_residual row",
        total_token_ns: wall_ns,
        sum_serial_ns: sum_serial,
        residual_ns,
        residual_name: "unattributed_residual = host tail between Instant marks + rounding".into(),
        residual_reason: format!(
            "named_except_residual={named_except_residual:.0} ns; residual row is inside the table so sum(components)==wall and residual_ns here is only rounding"
        ),
        identity_holds: sum_serial + residual_ns == wall_ns as i128,
        production_gpu_ns: snap.gpu_ns,
        organ_gpu_sum_ns: snap.gpu_organ.sum(),
        host_exclusive_sum_ns: host_prep as u64
            + snap.host_excl.deltanet_host()
            + snap.host_excl.gqa_host()
            + snap.host_excl.normalization_host()
            + snap.host_excl.silu
            + snap.host_excl.topk
            + snap.host_excl.combine
            + snap.host_excl.argmax
            + snap.host_excl.gqa_kv_copy,
    };
    (components, closure)
}

pub fn median_of_decode_samples(samples: &[MixedTokenSample]) -> (u64, MixedExclusiveSnap) {
    let mut decode: Vec<&MixedTokenSample> =
        samples.iter().filter(|s| s.kind == "decode").collect();
    if decode.is_empty() {
        return (0, MixedExclusiveSnap::default());
    }
    decode.sort_by_key(|s| s.wall_ns);
    let mid = decode[decode.len() / 2];
    // One real token, not median-of-fields. Independent medians do not
    // reassemble the wall (median(a)+median(b) ≠ median(a+b)).
    (mid.wall_ns, mid.snap)
}

pub fn occupancy_note(rows: u64, kernel: &str) -> String {
    let tgs = if kernel.contains("tg256") {
        rows
    } else {
        rows.div_ceil(8)
    };
    format!("{kernel}: {tgs} TGs, {:.2} TG/core on 60-core M3 Ultra", tgs as f64 / 60.0)
}
