const PQ_HEADER_BYTES: u64 = 64;
const PQ_TAIL_PAD_BYTES: u64 = 4;
#[derive(Clone, Copy)]
struct PqProjection {
    payload_bytes: u64,
    codebook_bytes: u64,
    elements: u64,
    sub: u64,
    touches: u64,
}
impl PqProjection {
    fn code_bytes(self) -> u64 {
        self.payload_bytes - PQ_HEADER_BYTES - self.codebook_bytes
    }
    fn resident_bytes(self) -> u64 {
        (self.codebook_bytes + self.code_bytes() + PQ_TAIL_PAD_BYTES) * self.touches
    }
    fn codebook_requests(self) -> u64 {
        self.elements * 2 * self.touches
    }
    fn code_requests(self) -> u64 {
        (self.elements / self.sub) * 4 * self.touches
    }
}
fn r4(payload_bytes: u64, elements: u64, touches: u64) -> PqProjection {
    PqProjection {
        payload_bytes,
        codebook_bytes: 16_384,
        elements,
        sub: 32,
        touches,
    }
}
fn sum_by(projections: &[PqProjection], f: impl Fn(PqProjection) -> u64) -> u64 {
    projections.iter().copied().map(f).sum()
}
#[test]
fn math_preserve_fixed_projection_census_is_exact() {
    let attention = [
        r4(409_664, 12_582_912, 78),
        r4(1_065_024, 33_554_432, 78),
        r4(127_040, 3_538_944, 78),
        r4(475_200, 14_680_064, 78),
        r4(3_162_176, 100_663_296, 78),
    ];
    let dense = [
        r4(2_375_744, 75_497_472, 3),
        r4(2_375_744, 75_497_472, 3),
        r4(2_375_744, 75_497_472, 3),
    ];
    let shared = [
        r4(409_664, 12_582_912, 75),
        r4(409_664, 12_582_912, 75),
        r4(409_664, 12_582_912, 75),
    ];
    assert_eq!(
        attention.map(PqProjection::resident_bytes),
        [31_949_112, 83_067_192, 9_904_440, 37_060_920, 246_645_048]
    );
    assert_eq!(
        sum_by(&attention, PqProjection::resident_bytes),
        408_626_712
    );
    assert_eq!(sum_by(&dense, PqProjection::resident_bytes), 21_381_156);
    assert_eq!(sum_by(&shared, PqProjection::resident_bytes), 92_160_900);
    let fixed_pq: Vec<_> = attention.into_iter().chain(dense).chain(shared).collect();
    assert_eq!(sum_by(&fixed_pq, PqProjection::resident_bytes), 522_168_768);
    assert_eq!(
        sum_by(&fixed_pq, PqProjection::codebook_requests),
        32_764_329_984
    );
    assert_eq!(
        sum_by(&fixed_pq, PqProjection::code_requests),
        2_047_770_624
    );
    let indexer_bf16 = (16_777_216 + 1_572_864 + 393_216) * 21;
    let router_bf16 = 3_145_728 * 75;
    let head_bf16 = 154_880 * 6_144 * 2;
    let fixed_native_bf16 = indexer_bf16 + router_bf16 + head_bf16;
    assert_eq!(fixed_native_bf16, 2_532_704_256);
    assert_eq!(
        sum_by(&fixed_pq, PqProjection::resident_bytes) + fixed_native_bf16,
        3_054_873_024
    );
}
#[test]
fn math_preserve_routed_census_is_route_conditioned_and_bounded() {
    const R0_BY_LAYER_3_TO_77: [u64; 75] = [
        7, 1, 2, 9, 3, 11, 12, 9, 2, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 7, 0, 9,
        1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 3, 0, 6, 1, 3, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1,
        2, 0, 0, 1, 0, 0, 6, 9, 1, 0, 0, 0, 2, 0, 9,
    ];
    let r0_candidates: u64 = R0_BY_LAYER_3_TO_77.iter().sum();
    let native_candidates: u64 = R0_BY_LAYER_3_TO_77.iter().map(|r0| 13 - r0).sum();
    let r4_candidates = 243 * 75;
    assert_eq!(
        (r4_candidates, native_candidates, r0_candidates),
        (18_225, 844, 131)
    );
    const R4_TRIPLET_RESIDENT: u64 = 1_228_812;
    const R0_TRIPLET_RESIDENT: u64 = 4_134_924;
    const NATIVE_TRIPLET_RESIDENT: u64 = 75_497_472;
    const R4_TRIPLET_REQUESTS: u64 = 80_216_064;
    const R0_TRIPLET_REQUESTS: u64 = 94_371_840;
    const NATIVE_TRIPLET_REQUESTS: u64 = 75_497_472;
    let route_min = 75 * 8 * R4_TRIPLET_RESIDENT;
    assert_eq!(route_min, 737_287_200);
    let mut max_native = 0u64;
    let mut max_r0 = 0u64;
    for &r0 in &R0_BY_LAYER_3_TO_77 {
        let native = 13 - r0;
        let selected_native = native.min(8);
        max_native += selected_native;
        max_r0 += r0.min(8 - selected_native);
    }
    assert_eq!((max_native, max_r0), (561, 39));
    let route_max = max_native * NATIVE_TRIPLET_RESIDENT + max_r0 * R0_TRIPLET_RESIDENT;
    assert_eq!(route_max, 42_515_343_828);
    let fixed = 3_054_873_024;
    assert_eq!(fixed + route_min, 3_792_160_224);
    assert_eq!(fixed + route_max, 45_570_216_852);
    let min_request_route = 561 * NATIVE_TRIPLET_REQUESTS + 39 * R4_TRIPLET_REQUESTS;
    let max_request_route = 119 * R0_TRIPLET_REQUESTS + 481 * R4_TRIPLET_REQUESTS;
    assert_eq!(min_request_route, 45_482_508_288);
    assert_eq!(max_request_route, 49_814_175_744);
    let fixed_requests = 37_344_804_864;
    assert_eq!(fixed_requests + min_request_route, 82_827_313_152);
    assert_eq!(fixed_requests + max_request_route, 87_158_980_608);
}
#[test]
fn r0_geometry_is_only_a_historical_control() {
    let legacy_geometry = 8u64 * 3 * 1_378_368 * 78;
    assert_eq!(legacy_geometry, 2_580_304_896);
    let historical_fixed = 522_168_768u64 + 2 * 2_532_704_256;
    assert_eq!(historical_fixed, 5_587_577_280);
    let historical_all_r0_75_layer_control = historical_fixed + 600 * 4_134_924;
    assert_eq!(historical_all_r0_75_layer_control, 8_068_531_680);
}
