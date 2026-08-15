// Unregistered static ABI draft for a future Qwen80 GQA K/V cache stage.
// This file is intentionally not listed in metal/mod.rs and is not dispatched
// by any runtime.  The matching Rust target inspects it as source text only.

#include <metal_stdlib>
using namespace metal;

struct Qwen80GqaKvCachePreflightParams {
    uint current_position;
    uint max_seq_len;
    uint kv_heads;
    uint head_dim;
};

// Static host ABI for the exact compact direct-packed Q/K/V/O projection
// family of source layer 3. This draft deliberately has no tensor payload
// pointers and no decoder integration: a future lease-gated component host
// must bind admitted sign-bit plus FP16-group-scale payloads before it can
// populate this ABI or dispatch a projection kernel.
struct Qwen80GqaCompactDirectPackedPayloadAbi {
    uint q_projection_rows;
    uint k_projection_rows;
    uint v_projection_rows;
    uint o_projection_rows;
    uint hidden_size;
    uint query_dimension;
    uint direct_packed_group_size;
    uint selected_gqa_layer;
    uint selected_gqa_slot;
};

// Host-encoded ABI for the reusable component child.  A future upstream owns
// one real TCB and supplies the source-bound [2048] hidden input plus these
// caller-owned active/rollback cache domains.  This static source neither
// decodes compact payloads nor authorizes an encoder/dispatch on its own.
struct Qwen80GqaComponentChildTcbAbi {
    uint hidden_width;
    uint current_position;
    uint max_seq_len;
    uint selected_gqa_layer;
    uint selected_gqa_slot;
    uint append_before_causal_read;
    uint active_and_rollback_disjoint;
};

// Future command order must encode this append before the causal-mask/read
// stages. `row_index` spans exactly KV_HEADS * HEAD_DIM for one selected slot.
kernel void qwen80_gqa_kv_cache_append_preflight(
    device const float *key_row [[buffer(0)]],
    device const float *value_row [[buffer(1)]],
    device float *key_cache [[buffer(2)]],
    device float *value_cache [[buffer(3)]],
    constant Qwen80GqaKvCachePreflightParams &params [[buffer(4)]],
    uint row_index [[thread_position_in_grid]]) {
    const uint row_elements = params.kv_heads * params.head_dim;
    if (params.current_position >= params.max_seq_len || row_index >= row_elements) {
        return;
    }
    const uint cache_index = params.current_position * row_elements + row_index;
    key_cache[cache_index] = key_row[row_index];
    value_cache[cache_index] = value_row[row_index];
}

// Future positions receive -infinity.  A future attention stage may only read
// slots whose mask is zero, so current position is included after append.
kernel void qwen80_gqa_causal_mask_preflight(
    device float *causal_mask [[buffer(0)]],
    constant Qwen80GqaKvCachePreflightParams &params [[buffer(1)]],
    uint position [[thread_position_in_grid]]) {
    if (position >= params.max_seq_len) {
        return;
    }
    causal_mask[position] = position <= params.current_position ? 0.0f : -INFINITY;
}

// A future abort/rollback stage restores only the selected K/V slot from a
// snapshot acquired before append. It has no implicit commit behavior.
kernel void qwen80_gqa_kv_cache_rollback_preflight(
    device const float *rollback_key_cache [[buffer(0)]],
    device const float *rollback_value_cache [[buffer(1)]],
    device float *active_key_cache [[buffer(2)]],
    device float *active_value_cache [[buffer(3)]],
    constant Qwen80GqaKvCachePreflightParams &params [[buffer(4)]],
    uint row_index [[thread_position_in_grid]]) {
    const uint row_elements = params.kv_heads * params.head_dim;
    if (params.current_position >= params.max_seq_len || row_index >= row_elements) {
        return;
    }
    const uint cache_index = params.current_position * row_elements + row_index;
    active_key_cache[cache_index] = rollback_key_cache[cache_index];
    active_value_cache[cache_index] = rollback_value_cache[cache_index];
}

// Future readback contract for exactly one selected session/layer/slot.  It
// copies only the current synthetic cache row and its causal-mask scalar into
// host-owned readback buffers.  This source is unregistered and static-only;
// no host runtime is permitted to dispatch it from this preflight target.
kernel void qwen80_gqa_component_readback_preflight(
    device const float *active_key_cache [[buffer(0)]],
    device const float *active_value_cache [[buffer(1)]],
    device const float *causal_mask [[buffer(2)]],
    device float *key_readback [[buffer(3)]],
    device float *value_readback [[buffer(4)]],
    device float *mask_readback [[buffer(5)]],
    constant Qwen80GqaKvCachePreflightParams &params [[buffer(6)]],
    constant Qwen80GqaCompactDirectPackedPayloadAbi &payload_abi [[buffer(7)]],
    uint row_index [[thread_position_in_grid]]) {
    const uint row_elements = params.kv_heads * params.head_dim;
    if (params.current_position >= params.max_seq_len || row_index >= row_elements) {
        return;
    }
    // The static source-shaped ABI fixes layer 3 / slot 0 and direct-packed
    // group size 128. A future host must reject mismatch before encode.
    if (payload_abi.direct_packed_group_size != 128u ||
        payload_abi.selected_gqa_layer != 3u ||
        payload_abi.selected_gqa_slot != 0u) {
        return;
    }
    const uint cache_index = params.current_position * row_elements + row_index;
    key_readback[row_index] = active_key_cache[cache_index];
    value_readback[row_index] = active_value_cache[cache_index];
    if (row_index == 0u) {
        mask_readback[0] = causal_mask[params.current_position];
    }
}
