// Exact-token sequential llama.cpp oracle for Hawking K0 bisection.
//
// Build on a llama.cpp development install, for example:
//   c++ -O3 -std=c++17 tools/llama_sequential_oracle.cpp \
//     -I$(brew --prefix ggml)/include -L$(brew --prefix ggml)/lib $(pkg-config --cflags --libs llama) \
//     -Wl,-rpath,$(brew --prefix llama.cpp)/lib -Wl,-rpath,$(brew --prefix ggml)/lib \
//     -o target/release/llama-sequential-oracle
//
// This deliberately emits only scalar logit sums and the greedy id. It does
// not replace the eval-callback tensor oracle; it narrows a batched prefill
// mismatch to its first exact token position without retaining model tensors.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <ggml-backend.h>
#include <ggml-alloc.h>
#include <ggml.h>
#include <llama.h>

namespace {

constexpr const char * kSchema = "hawking.tg.llama_sequential_oracle.v1";
constexpr const char * kDefaultMetalPlugin = "/opt/homebrew/Cellar/ggml/0.13.1/libexec/libggml-metal.so";

struct FlashAttentionReplay {
    bool requested = false;
    bool attempted = false;
    bool completed = false;
    bool exact_f32 = false;
    const char * error = "none";
    size_t elements = 0;
    double max_abs_error = 0.0;
    double l1_error = 0.0;
    struct TensorLayout {
        bool captured = false;
        int type = 0;
        int64_t ne[GGML_MAX_DIMS] = {};
        size_t nb[GGML_MAX_DIMS] = {};
    } source_layouts[4];
    int32_t precision = 0;
};

struct CheckpointCapture {
    const char * name = nullptr;
    enum ggml_op required_op = GGML_OP_NONE;
    bool list_checkpoints = false;
    int checkpoint_input_index = -1;
    size_t f16_offset = 0;
    size_t f16_count = 0;
    bool captured = false;
    bool f32 = false;
    const char * value_source = "none";
    std::vector<float> values;
    std::vector<std::string> * catalog = nullptr;
    FlashAttentionReplay flash_attention_replay;
    bool rope_params_captured = false;
    int32_t rope_params[15] = {};
};

void usage(const char * program) {
    std::fprintf(
        stderr,
        "usage: %s --model MODEL (--prompt PROMPT | --prompt-file PATH) [--gpu-layers N] [--ctx-size N] "
        "[--checkpoint TENSOR_NAME [--checkpoint-op rope] "
        "[--checkpoint-f16-offset ELEMENT --checkpoint-f16-count ELEMENTS "
        "[--checkpoint-f16-current-token-stride ELEMENTS] "
        "[--checkpoint-input-index 0..4] [--replay-fattn]] "
        "[--list-checkpoints] [--tokenize-only] [--measure-warmup N --measure-tokens N]\n",
        program);
}

bool parse_i32(const char * text, int32_t * value) {
    char * end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < std::numeric_limits<int32_t>::min() ||
        parsed > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    *value = static_cast<int32_t>(parsed);
    return true;
}

bool parse_size(const char * text, size_t * value) {
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    *value = static_cast<size_t>(parsed);
    return true;
}

void discard_llama_log(enum ggml_log_level, const char *, void *) {}

void write_json_float(std::ostream & stream, float value) {
    if (std::isfinite(value)) {
        stream << value;
    } else if (std::isnan(value)) {
        stream << "\"nan\"";
    } else {
        stream << (std::signbit(value) ? "\"-inf\"" : "\"inf\"");
    }
}

void write_flash_attention_replay(
        std::ostream & stream,
        const FlashAttentionReplay & replay) {
    stream << "{\"attempted\": " << (replay.attempted ? "true" : "false")
           << ", \"completed\": " << (replay.completed ? "true" : "false")
           << ", \"exact_f32\": " << (replay.exact_f32 ? "true" : "false")
           << ", \"error\": \"" << replay.error << "\""
           << ", \"elements\": " << replay.elements
           << ", \"max_abs_error\": " << replay.max_abs_error
           << ", \"l1_error\": " << replay.l1_error
           << ", \"precision\": " << replay.precision
           << ", \"source_layouts\": [";
    for (int source = 0; source < 4; ++source) {
        if (source != 0) {
            stream << ", ";
        }
        const auto & layout = replay.source_layouts[source];
        stream << "{\"captured\": " << (layout.captured ? "true" : "false")
               << ", \"type\": " << layout.type << ", \"ne\": [";
        for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
            if (dimension != 0) {
                stream << ", ";
            }
            stream << layout.ne[dimension];
        }
        stream << "], \"nb\": [";
        for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
            if (dimension != 0) {
                stream << ", ";
            }
            stream << layout.nb[dimension];
        }
        stream << "]}";
    }
    stream << "]}";
}

void capture_tensor_layout(
        FlashAttentionReplay::TensorLayout * destination,
        const struct ggml_tensor * source) {
    destination->captured = source != nullptr;
    if (source == nullptr) {
        return;
    }
    destination->type = static_cast<int>(source->type);
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        destination->ne[dimension] = source->ne[dimension];
        destination->nb[dimension] = source->nb[dimension];
    }
}

void write_rope_params(std::ostream & stream, const CheckpointCapture & capture) {
    stream << "{\"captured\": " << (capture.rope_params_captured ? "true" : "false");
    if (capture.rope_params_captured) {
        float freq_base = 0.0f;
        float freq_scale = 0.0f;
        float ext_factor = 0.0f;
        float attn_factor = 0.0f;
        float beta_fast = 0.0f;
        float beta_slow = 0.0f;
        std::memcpy(&freq_base, capture.rope_params + 5, sizeof(freq_base));
        std::memcpy(&freq_scale, capture.rope_params + 6, sizeof(freq_scale));
        std::memcpy(&ext_factor, capture.rope_params + 7, sizeof(ext_factor));
        std::memcpy(&attn_factor, capture.rope_params + 8, sizeof(attn_factor));
        std::memcpy(&beta_fast, capture.rope_params + 9, sizeof(beta_fast));
        std::memcpy(&beta_slow, capture.rope_params + 10, sizeof(beta_slow));
        stream << ", \"n_dims\": " << capture.rope_params[1]
               << ", \"mode\": " << capture.rope_params[2]
               << ", \"n_ctx_orig\": " << capture.rope_params[4]
               << ", \"freq_base\": " << freq_base
               << ", \"freq_scale\": " << freq_scale
               << ", \"ext_factor\": " << ext_factor
               << ", \"attn_factor\": " << attn_factor
               << ", \"beta_fast\": " << beta_fast
               << ", \"beta_slow\": " << beta_slow;
    }
    stream << "}";
}

struct ggml_tensor * clone_tensor_layout(
        struct ggml_context * context,
        const struct ggml_tensor * source) {
    const size_t type_size = ggml_type_size(source->type);
    const size_t bytes = ggml_nbytes(source);
    if (type_size == 0 || bytes % type_size != 0) {
        return nullptr;
    }
    // Allocate the full byte extent of the view, then retain the original
    // logical dimensions and strides. FATTN receives cache views directly;
    // flattening those views changes the source layout before the authority
    // primitive ever runs.
    ggml_tensor * const destination = ggml_new_tensor_1d(
        context, source->type, static_cast<int64_t>(bytes / type_size));
    if (destination == nullptr) {
        return nullptr;
    }
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        destination->ne[dimension] = source->ne[dimension];
        destination->nb[dimension] = source->nb[dimension];
    }
    return destination;
}

bool replay_flash_attention(
        const struct ggml_tensor * authority,
        FlashAttentionReplay * replay) {
    replay->attempted = true;
    if (authority->op != GGML_OP_FLASH_ATTN_EXT) {
        replay->error = "checkpoint is not FLASH_ATTN_EXT";
        return false;
    }
    if (authority->src[0] == nullptr || authority->src[1] == nullptr ||
        authority->src[2] == nullptr || authority->src[3] == nullptr) {
        replay->error = "FLASH_ATTN_EXT is missing a required source";
        return false;
    }
    if (authority->src[4] != nullptr) {
        replay->error = "FLASH_ATTN_EXT sink replay is not implemented";
        return false;
    }
    const ggml_tensor * const q_source = authority->src[0];
    const ggml_tensor * const k_source = authority->src[1];
    const ggml_tensor * const v_source = authority->src[2];
    const ggml_tensor * const mask_source = authority->src[3];
    capture_tensor_layout(&replay->source_layouts[0], q_source);
    capture_tensor_layout(&replay->source_layouts[1], k_source);
    capture_tensor_layout(&replay->source_layouts[2], v_source);
    capture_tensor_layout(&replay->source_layouts[3], mask_source);
    replay->precision = static_cast<int32_t>(ggml_flash_attn_ext_get_prec(authority));
    if (q_source->type != GGML_TYPE_F32 || k_source->type != GGML_TYPE_F16 ||
        v_source->type != GGML_TYPE_F16 || mask_source->type != GGML_TYPE_F16 ||
        !ggml_is_contiguous(mask_source)) {
        replay->error = "FLASH_ATTN_EXT sources are not f32/f16/f16/contiguous-f16";
        return false;
    }

    ggml_backend_reg_t metal_registry = ggml_backend_reg_by_name("Metal");
    if (metal_registry == nullptr) {
        const char * const plugin_path = std::getenv("HAWKING_GGML_METAL_PLUGIN");
        metal_registry = ggml_backend_load(
            plugin_path != nullptr ? plugin_path : kDefaultMetalPlugin);
    }
    if (metal_registry == nullptr || ggml_backend_reg_dev_count(metal_registry) == 0) {
        replay->error = "ggml Metal plugin load failed";
        return false;
    }
    ggml_backend_t backend = ggml_backend_dev_init(
        ggml_backend_reg_dev_get(metal_registry, 0), nullptr);
    if (backend == nullptr) {
        replay->error = "ggml Metal backend initialization failed";
        return false;
    }
    ggml_context * tensor_context = nullptr;
    ggml_context * graph_context = nullptr;
    ggml_backend_buffer_t input_buffer = nullptr;
    ggml_gallocr_t graph_allocator = nullptr;
    bool success = false;
    do {
        const ggml_init_params tensor_params = {
            /*.mem_size   =*/ 4 * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        tensor_context = ggml_init(tensor_params);
        if (tensor_context == nullptr) {
            replay->error = "ggml input context allocation failed";
            break;
        }
        ggml_tensor * const q = clone_tensor_layout(tensor_context, q_source);
        ggml_tensor * const k = clone_tensor_layout(tensor_context, k_source);
        ggml_tensor * const v = clone_tensor_layout(tensor_context, v_source);
        ggml_tensor * const mask = clone_tensor_layout(tensor_context, mask_source);
        if (q == nullptr || k == nullptr || v == nullptr || mask == nullptr) {
            replay->error = "ggml input tensor allocation failed";
            break;
        }
        input_buffer = ggml_backend_alloc_ctx_tensors(tensor_context, backend);
        if (input_buffer == nullptr) {
            replay->error = "ggml Metal input buffer allocation failed";
            break;
        }
        std::vector<unsigned char> q_bytes(ggml_nbytes(q_source));
        std::vector<unsigned char> k_bytes(ggml_nbytes(k_source));
        std::vector<unsigned char> v_bytes(ggml_nbytes(v_source));
        std::vector<unsigned char> mask_bytes(ggml_nbytes(mask_source));
        ggml_backend_tensor_get(q_source, q_bytes.data(), 0, q_bytes.size());
        ggml_backend_tensor_get(k_source, k_bytes.data(), 0, k_bytes.size());
        ggml_backend_tensor_get(v_source, v_bytes.data(), 0, v_bytes.size());
        ggml_backend_tensor_get(mask_source, mask_bytes.data(), 0, mask_bytes.size());
        ggml_backend_tensor_set(q, q_bytes.data(), 0, q_bytes.size());
        ggml_backend_tensor_set(k, k_bytes.data(), 0, k_bytes.size());
        ggml_backend_tensor_set(v, v_bytes.data(), 0, v_bytes.size());
        ggml_backend_tensor_set(mask, mask_bytes.data(), 0, mask_bytes.size());

        const ggml_init_params graph_params = {
            /*.mem_size   =*/ 2 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        graph_context = ggml_init(graph_params);
        if (graph_context == nullptr) {
            replay->error = "ggml replay graph context allocation failed";
            break;
        }
        float scale = 0.0f;
        float max_bias = 0.0f;
        float logit_softcap = 0.0f;
        std::memcpy(&scale, authority->op_params + 0 * sizeof(float), sizeof(scale));
        std::memcpy(&max_bias, authority->op_params + 1 * sizeof(float), sizeof(max_bias));
        std::memcpy(&logit_softcap, authority->op_params + 2 * sizeof(float), sizeof(logit_softcap));
        ggml_tensor * const result = ggml_flash_attn_ext(
            graph_context, q, k, v, mask, scale, max_bias, logit_softcap);
        ggml_flash_attn_ext_set_prec(
            result, ggml_flash_attn_ext_get_prec(authority));
        // The graph visitor tracks the result plus the four input leaves.
        // Leave enough hash capacity for those parents even though this
        // diagnostic has one compute node.
        ggml_cgraph * const graph = ggml_new_graph_custom(graph_context, 16, false);
        ggml_build_forward_expand(graph, result);
        graph_allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (graph_allocator == nullptr || !ggml_gallocr_alloc_graph(graph_allocator, graph)) {
            replay->error = "ggml replay graph allocation failed";
            break;
        }
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            replay->error = "ggml Metal replay compute failed";
            break;
        }
        std::vector<float> replay_values(ggml_nelements(result));
        std::vector<float> authority_values(ggml_nelements(authority));
        ggml_backend_tensor_get(result, replay_values.data(), 0, ggml_nbytes(result));
        ggml_backend_tensor_get(authority, authority_values.data(), 0, ggml_nbytes(authority));
        if (replay_values.size() != authority_values.size()) {
            replay->error = "FLASH_ATTN_EXT replay output shape differs";
            break;
        }
        replay->elements = replay_values.size();
        replay->exact_f32 = std::memcmp(
            replay_values.data(), authority_values.data(), ggml_nbytes(authority)) == 0;
        for (size_t index = 0; index < replay_values.size(); ++index) {
            const double difference = std::abs(
                static_cast<double>(replay_values[index]) - static_cast<double>(authority_values[index]));
            replay->max_abs_error = std::max(replay->max_abs_error, difference);
            replay->l1_error += difference;
        }
        replay->completed = true;
        success = true;
    } while (false);
    if (graph_allocator != nullptr) {
        ggml_gallocr_free(graph_allocator);
    }
    if (graph_context != nullptr) {
        ggml_free(graph_context);
    }
    if (input_buffer != nullptr) {
        ggml_backend_buffer_free(input_buffer);
    }
    if (tensor_context != nullptr) {
        ggml_free(tensor_context);
    }
    ggml_backend_free(backend);
    return success;
}

bool capture_checkpoint(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto * capture = static_cast<CheckpointCapture *>(user_data);
    if (ask) {
        if (capture->list_checkpoints && capture->catalog != nullptr) {
            const std::string descriptor = std::string(tensor->name) + "|" +
                                           ggml_type_name(tensor->type) + "|" +
                                           ggml_op_name(tensor->op) + "|" +
                                           std::to_string(ggml_nelements(tensor)) + "|" +
                                           std::to_string(tensor->ne[0]) + "x" +
                                           std::to_string(tensor->ne[1]) + "x" +
                                           std::to_string(tensor->ne[2]) + "x" +
                                           std::to_string(tensor->ne[3]);
            if (std::find(capture->catalog->begin(), capture->catalog->end(), descriptor) ==
                capture->catalog->end()) {
                capture->catalog->push_back(descriptor);
            }
        }
        return capture->name != nullptr && std::strcmp(tensor->name, capture->name) == 0 &&
               (capture->required_op == GGML_OP_NONE || tensor->op == capture->required_op);
    }
    if (capture->flash_attention_replay.requested &&
        !capture->flash_attention_replay.attempted) {
        replay_flash_attention(tensor, &capture->flash_attention_replay);
    }
    if (tensor->op == GGML_OP_ROPE) {
        capture->rope_params_captured = true;
        std::memcpy(capture->rope_params, tensor->op_params, sizeof(capture->rope_params));
    }
    const ggml_tensor * value_tensor = tensor;
    if (capture->checkpoint_input_index >= 0) {
        value_tensor = tensor->src[capture->checkpoint_input_index];
        if (value_tensor == nullptr) {
            capture->captured = true;
            return true;
        }
    }
    if (!capture->captured && value_tensor->type == GGML_TYPE_F32) {
        const size_t bytes = ggml_nbytes(value_tensor);
        if (bytes % sizeof(float) == 0) {
            capture->values.resize(bytes / sizeof(float));
            ggml_backend_tensor_get(value_tensor, capture->values.data(), 0, bytes);
            capture->f32 = true;
            capture->value_source = "f32";
        }
    } else if (!capture->captured && value_tensor->type == GGML_TYPE_F16 && capture->f16_count > 0) {
        const size_t elements = static_cast<size_t>(ggml_nelements(value_tensor));
        if (capture->f16_offset < elements) {
            const size_t count = std::min(capture->f16_count, elements - capture->f16_offset);
            std::vector<ggml_fp16_t> source(count);
            ggml_backend_tensor_get(
                value_tensor, source.data(), capture->f16_offset * sizeof(ggml_fp16_t),
                count * sizeof(ggml_fp16_t));
            capture->values.resize(count);
            for (size_t index = 0; index < count; ++index) {
                capture->values[index] = ggml_fp16_to_fp32(source[index]);
            }
            // Values are emitted as f32 after a lossless f16 decode, so the
            // existing vector comparator can consume them without a second
            // conversion. `value_source` preserves the native storage type.
            capture->f32 = true;
            capture->value_source = "f16";
        }
    }
    capture->captured = true;
    // Returning false here would abort llama_decode().
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * prompt_arg = nullptr;
    const char * prompt_file = nullptr;
    const char * checkpoint_name = nullptr;
    const char * checkpoint_op = nullptr;
    bool list_checkpoints = false;
    bool tokenize_only = false;
    int32_t measure_warmup = 0;
    int32_t measure_tokens = 0;
    size_t checkpoint_f16_offset = 0;
    size_t checkpoint_f16_count = 0;
    size_t checkpoint_f16_current_token_stride = 0;
    int32_t checkpoint_input_index = -1;
    bool replay_fattn = false;
    int32_t gpu_layers = 0;
    int32_t ctx_size = 0;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if ((argument == "--model" || argument == "--prompt" || argument == "--prompt-file" || argument == "--gpu-layers" ||
             argument == "--ctx-size" || argument == "--checkpoint" || argument == "--checkpoint-op" ||
             argument == "--checkpoint-f16-offset" || argument == "--checkpoint-f16-count" ||
             argument == "--checkpoint-f16-current-token-stride" ||
             argument == "--checkpoint-input-index" || argument == "--measure-warmup" ||
             argument == "--measure-tokens") && index + 1 >= argc) {
            usage(argv[0]);
            return 2;
        }
        if (argument == "--model") {
            model_path = argv[++index];
        } else if (argument == "--prompt") {
            prompt_arg = argv[++index];
        } else if (argument == "--prompt-file") {
            prompt_file = argv[++index];
        } else if (argument == "--gpu-layers") {
            const std::string requested_layers(argv[++index]);
            // Keep this JSON oracle's selector compatible with the regular
            // llama.cpp front-end used by the scalar callback oracle.
            if (requested_layers == "all") {
                gpu_layers = 99;
            } else if (!parse_i32(requested_layers.c_str(), &gpu_layers)) {
                std::fprintf(stderr, "invalid --gpu-layers\n");
                return 2;
            }
        } else if (argument == "--ctx-size") {
            if (!parse_i32(argv[++index], &ctx_size) || ctx_size < 1) {
                std::fprintf(stderr, "invalid --ctx-size\n");
                return 2;
            }
        } else if (argument == "--checkpoint") {
            checkpoint_name = argv[++index];
        } else if (argument == "--checkpoint-op") {
            checkpoint_op = argv[++index];
        } else if (argument == "--checkpoint-f16-offset") {
            if (!parse_size(argv[++index], &checkpoint_f16_offset)) {
                std::fprintf(stderr, "invalid --checkpoint-f16-offset\n");
                return 2;
            }
        } else if (argument == "--checkpoint-f16-count") {
            if (!parse_size(argv[++index], &checkpoint_f16_count) || checkpoint_f16_count == 0) {
                std::fprintf(stderr, "invalid --checkpoint-f16-count\n");
                return 2;
            }
        } else if (argument == "--checkpoint-f16-current-token-stride") {
            if (!parse_size(argv[++index], &checkpoint_f16_current_token_stride) ||
                checkpoint_f16_current_token_stride == 0) {
                std::fprintf(stderr, "invalid --checkpoint-f16-current-token-stride\n");
                return 2;
            }
        } else if (argument == "--checkpoint-input-index") {
            if (!parse_i32(argv[++index], &checkpoint_input_index) || checkpoint_input_index < 0 ||
                checkpoint_input_index > 4) {
                std::fprintf(stderr, "invalid --checkpoint-input-index (requires 0..4)\n");
                return 2;
            }
        } else if (argument == "--replay-fattn") {
            replay_fattn = true;
        } else if (argument == "--list-checkpoints") {
            list_checkpoints = true;
        } else if (argument == "--tokenize-only") {
            tokenize_only = true;
        } else if (argument == "--measure-warmup") {
            if (!parse_i32(argv[++index], &measure_warmup) || measure_warmup < 0) {
                std::fprintf(stderr, "invalid --measure-warmup\n");
                return 2;
            }
        } else if (argument == "--measure-tokens") {
            if (!parse_i32(argv[++index], &measure_tokens) || measure_tokens < 1) {
                std::fprintf(stderr, "invalid --measure-tokens\n");
                return 2;
            }
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (model_path == nullptr || (prompt_arg == nullptr && prompt_file == nullptr) ||
        (prompt_arg != nullptr && prompt_file != nullptr)) {
        usage(argv[0]);
        return 2;
    }
    if ((checkpoint_f16_offset != 0 || checkpoint_f16_count != 0 ||
         checkpoint_f16_current_token_stride != 0) && checkpoint_name == nullptr) {
        std::fprintf(stderr, "f16 slice options require --checkpoint\n");
        return 2;
    }
    if (checkpoint_input_index >= 0 && checkpoint_name == nullptr) {
        std::fprintf(stderr, "--checkpoint-input-index requires --checkpoint\n");
        return 2;
    }
    if (replay_fattn && checkpoint_name == nullptr) {
        std::fprintf(stderr, "--replay-fattn requires --checkpoint\n");
        return 2;
    }
    if ((measure_warmup != 0 || measure_tokens != 0) &&
        (measure_warmup < 1 || measure_tokens < 1)) {
        std::fprintf(stderr, "--measure-warmup and --measure-tokens must be used together\n");
        return 2;
    }
    if (measure_tokens != 0 && (checkpoint_name != nullptr || list_checkpoints || replay_fattn)) {
        std::fprintf(stderr, "measurement mode cannot be combined with checkpoint capture\n");
        return 2;
    }

    std::string prompt_storage;
    if (prompt_file != nullptr) {
        std::ifstream input(prompt_file, std::ios::binary);
        if (!input) {
            std::fprintf(stderr, "cannot read prompt file: %s\n", prompt_file);
            return 2;
        }
        prompt_storage.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    } else {
        prompt_storage = prompt_arg;
    }

    // llama.cpp otherwise writes model-loading diagnostics to stdout. This
    // process is a JSON protocol endpoint, so retain its errors from this
    // wrapper while discarding library logging.
    llama_log_set(discard_llama_log, nullptr);
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", model_path);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t prompt_bytes = static_cast<int32_t>(prompt_storage.size());
    int32_t token_count = llama_tokenize(vocab, prompt_storage.data(), prompt_bytes, nullptr, 0, true, false);
    if (token_count >= 0) {
        std::fprintf(stderr, "llama_tokenize did not report a required token count\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    token_count = -token_count;
    std::vector<llama_token> tokens(static_cast<size_t>(token_count));
    if (llama_tokenize(vocab, prompt_storage.data(), prompt_bytes, tokens.data(), token_count, true, false) != token_count) {
        std::fprintf(stderr, "llama_tokenize failed to encode prompt\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    if (tokenize_only) {
        std::cout << "{\"schema\": \"" << kSchema << "\", \"prompt_bytes\": "
                  << prompt_bytes << ", \"bos_token_id\": " << llama_vocab_bos(vocab)
                  << ", \"eos_token_id\": " << llama_vocab_eos(vocab)
                  << ", \"prompt_token_ids\": [";
        for (int32_t index = 0; index < token_count; ++index) {
            if (index != 0) {
                std::cout << ", ";
            }
            std::cout << tokens[static_cast<size_t>(index)];
        }
        std::cout << "]}\n";
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    llama_context_params context_params = llama_context_default_params();
    const int32_t required_ctx = token_count + measure_warmup + measure_tokens;
    context_params.n_ctx = ctx_size > 0 ? static_cast<uint32_t>(ctx_size)
                                        : static_cast<uint32_t>(required_ctx);
    if (context_params.n_ctx < static_cast<uint32_t>(required_ctx)) {
        std::fprintf(stderr, "--ctx-size is too small for prompt + warmup + measurement\n");
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    context_params.n_batch = 1;
    context_params.n_ubatch = 1;
    context_params.no_perf = true;
    CheckpointCapture checkpoint;
    CheckpointCapture first_checkpoint;
    std::vector<CheckpointCapture> checkpoint_records;
    checkpoint.name = checkpoint_name;
    checkpoint.f16_offset = checkpoint_f16_offset;
    checkpoint.f16_count = checkpoint_f16_count;
    checkpoint.list_checkpoints = list_checkpoints;
    checkpoint.checkpoint_input_index = checkpoint_input_index;
    checkpoint.flash_attention_replay.requested = replay_fattn;
    std::vector<std::string> checkpoint_catalog;
    checkpoint.catalog = &checkpoint_catalog;
    if (checkpoint_op != nullptr) {
        if (std::strcmp(checkpoint_op, "rope") != 0) {
            std::fprintf(stderr, "unsupported --checkpoint-op: %s\n", checkpoint_op);
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
        checkpoint.required_op = GGML_OP_ROPE;
    }
    if (checkpoint_name != nullptr || list_checkpoints) {
        context_params.cb_eval = capture_checkpoint;
        context_params.cb_eval_user_data = &checkpoint;
    }
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::fprintf(stderr, "failed to create llama context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    llama_batch batch = llama_batch_init(1, 0, 1);
    if (measure_tokens != 0) {
        const auto greedy = [vocab_size](const float * logits) {
            int32_t best = 0;
            for (int32_t index = 1; index < vocab_size; ++index) {
                if (logits[index] > logits[best]) {
                    best = index;
                }
            }
            return best;
        };
        for (int32_t index = 0; index < token_count; ++index) {
            batch.n_tokens = 1;
            batch.token[0] = tokens[static_cast<size_t>(index)];
            batch.pos[0] = index;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            if (llama_decode(context, batch) != 0) {
                std::fprintf(stderr, "llama_decode failed at prompt position %d\n", index);
                llama_batch_free(batch);
                llama_free(context);
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
        }
        const int32_t total_steps = measure_warmup + measure_tokens;
        std::vector<int32_t> measured_ids;
        std::vector<double> measured_ms;
        // Match Hawking's streamed-token boundary: greedy selection alone is
        // not a complete decode token if the competing runtime must also turn
        // its id into a text piece before handing it to the caller.
        std::string measured_text;
        measured_ids.reserve(static_cast<size_t>(measure_tokens));
        measured_ms.reserve(static_cast<size_t>(measure_tokens));
        const float * logits = llama_get_logits_ith(context, 0);
        int32_t last_id = tokens.back();
        for (int32_t step = 0; step < total_steps; ++step) {
            const bool measured = step >= measure_warmup;
            const auto started = std::chrono::steady_clock::now();
            if (step > 0) {
                batch.n_tokens = 1;
                batch.token[0] = last_id;
                batch.pos[0] = token_count + step - 1;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0] = 1;
                if (llama_decode(context, batch) != 0) {
                    std::fprintf(stderr, "llama_decode failed at decode position %d\n", step);
                    llama_batch_free(batch);
                    llama_free(context);
                    llama_model_free(model);
                    llama_backend_free();
                    return 1;
                }
                logits = llama_get_logits_ith(context, 0);
            }
            const int32_t next = greedy(logits);
            if (measured) {
                measured_ids.push_back(next);
                char piece[1024];
                int32_t piece_len = llama_token_to_piece(
                    vocab, next, piece, static_cast<int32_t>(sizeof(piece)), 0, true);
                if (piece_len < 0) {
                    std::vector<char> expanded(static_cast<size_t>(-piece_len));
                    piece_len = llama_token_to_piece(
                        vocab, next, expanded.data(), static_cast<int32_t>(expanded.size()), 0, true);
                    if (piece_len < 0) {
                        std::fprintf(stderr, "llama_token_to_piece failed for token %d\n", next);
                        llama_batch_free(batch);
                        llama_free(context);
                        llama_model_free(model);
                        llama_backend_free();
                        return 1;
                    }
                    measured_text.append(expanded.data(), static_cast<size_t>(piece_len));
                } else {
                    measured_text.append(piece, static_cast<size_t>(piece_len));
                }
            }
            last_id = next;
            if (measured) {
                measured_ms.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count());
            }
        }
        double total_ms = 0.0;
        for (const double sample : measured_ms) {
            total_ms += sample;
        }
        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
        std::cout << "{\"schema\": \"hawking.tg.llama_matched_decode.v1\", \"prompt_token_ids\": [";
        for (int32_t index = 0; index < token_count; ++index) {
            if (index != 0) std::cout << ", ";
            std::cout << tokens[static_cast<size_t>(index)];
        }
        std::cout << "], \"bos_token_id\": " << llama_vocab_bos(vocab)
                  << ", \"eos_token_id\": " << llama_vocab_eos(vocab)
                  << ", \"warmup_tokens\": " << measure_warmup
                  << ", \"generated_token_ids\": [";
        for (size_t index = 0; index < measured_ids.size(); ++index) {
            if (index != 0) std::cout << ", ";
            std::cout << measured_ids[index];
        }
        std::cout << "], \"decode_token_ms\": [";
        for (size_t index = 0; index < measured_ms.size(); ++index) {
            if (index != 0) std::cout << ", ";
            std::cout << measured_ms[index];
        }
        std::cout << "], \"decode_ms\": " << total_ms
                  << ", \"decode_tps\": "
                  << (static_cast<double>(measure_tokens) / (total_ms / 1000.0))
                  << ", \"decoded_bytes\": " << measured_text.size() << "}\n";
        llama_batch_free(batch);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }
    // This is a numerical-oracle protocol. The default iostream precision is
    // six significant digits, which can manufacture a several-ULP apparent
    // mismatch in a checkpoint vector before Hawking is even compared.
    // Keep both captured f32 vectors and the accumulated f64 logits sum
    // lossless in the evidence receipt.  Double precision covers f32 too.
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::cout << "{\n  \"schema\": \"" << kSchema << "\",\n  \"prompt_token_ids\": [";
    for (int32_t index = 0; index < token_count; ++index) {
        if (index != 0) {
            std::cout << ", ";
        }
        std::cout << tokens[static_cast<size_t>(index)];
    }
    std::cout << "],\n  \"records\": [\n";
    for (int32_t index = 0; index < token_count; ++index) {
        if (checkpoint_name != nullptr) {
            checkpoint.captured = false;
            checkpoint.f32 = false;
            checkpoint.value_source = "none";
            checkpoint.values.clear();
            checkpoint.rope_params_captured = false;
            std::memset(checkpoint.rope_params, 0, sizeof(checkpoint.rope_params));
            checkpoint.f16_offset = checkpoint_f16_offset +
                                    static_cast<size_t>(index) * checkpoint_f16_current_token_stride;
            checkpoint.flash_attention_replay = FlashAttentionReplay{};
            checkpoint.flash_attention_replay.requested = replay_fattn;
        }
        batch.n_tokens = 1;
        batch.token[0] = tokens[static_cast<size_t>(index)];
        batch.pos[0] = index;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        if (llama_decode(context, batch) != 0) {
            std::fprintf(stderr, "llama_decode failed at prompt position %d\n", index);
            llama_batch_free(batch);
            llama_free(context);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        const float * logits = llama_get_logits_ith(context, 0);
        if (logits == nullptr) {
            std::fprintf(stderr, "llama_get_logits_ith returned null at prompt position %d\n", index);
            llama_batch_free(batch);
            llama_free(context);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        double logits_sum = 0.0;
        int32_t greedy_token_id = 0;
        for (int32_t vocab_index = 0; vocab_index < vocab_size; ++vocab_index) {
            logits_sum += static_cast<double>(logits[vocab_index]);
            if (logits[vocab_index] > logits[greedy_token_id]) {
                greedy_token_id = vocab_index;
            }
        }
        if (checkpoint_name != nullptr) {
            if (index == 0) {
                first_checkpoint = checkpoint;
            }
            checkpoint_records.push_back(checkpoint);
        }
        if (index != 0) {
            std::cout << ",\n";
        }
        std::cout << "    {\"position\": " << index << ", \"token_id\": "
                  << tokens[static_cast<size_t>(index)] << ", \"logits_sum\": "
                  << logits_sum << ", \"greedy_token_id\": " << greedy_token_id << "}";
    }
    std::cout << "\n  ]";
    if (checkpoint_name != nullptr) {
        std::cout << ",\n  \"checkpoint\": {\"name\": \"" << checkpoint_name
                  << "\", \"captured\": " << (first_checkpoint.captured ? "true" : "false")
                  << ", \"f32\": " << (first_checkpoint.f32 ? "true" : "false")
                  << ", \"value_source\": \"" << first_checkpoint.value_source << "\""
                  << ", \"values\": [";
        for (size_t index = 0; index < first_checkpoint.values.size(); ++index) {
            if (index != 0) {
                std::cout << ", ";
            }
            write_json_float(std::cout, first_checkpoint.values[index]);
        }
        std::cout << "], \"position\": 0, \"rope_params\": ";
        write_rope_params(std::cout, first_checkpoint);
        std::cout << "}";
        if (replay_fattn) {
            const auto & replay = first_checkpoint.flash_attention_replay;
            std::cout << ",\n  \"flash_attention_replay\": ";
            write_flash_attention_replay(std::cout, replay);
        }
        std::cout << ",\n  \"checkpoint_records\": [";
        for (size_t record_index = 0; record_index < checkpoint_records.size(); ++record_index) {
            const auto & record = checkpoint_records[record_index];
            if (record_index != 0) {
                std::cout << ", ";
            }
            std::cout << "{\"position\": " << record_index
                      << ", \"captured\": " << (record.captured ? "true" : "false")
                      << ", \"f32\": " << (record.f32 ? "true" : "false")
                      << ", \"value_source\": \"" << record.value_source << "\""
                      << ", \"values\": [";
            for (size_t value_index = 0; value_index < record.values.size(); ++value_index) {
                if (value_index != 0) {
                    std::cout << ", ";
                }
                write_json_float(std::cout, record.values[value_index]);
            }
            std::cout << "]";
            if (checkpoint_op != nullptr && std::strcmp(checkpoint_op, "rope") == 0) {
                std::cout << ", \"rope_params\": ";
                write_rope_params(std::cout, record);
            }
            if (replay_fattn) {
                std::cout << ", \"flash_attention_replay\": ";
                write_flash_attention_replay(std::cout, record.flash_attention_replay);
            }
            std::cout << "}";
        }
        std::cout << "]";
    }
    if (list_checkpoints) {
        std::cout << ",\n  \"checkpoint_catalog\": [";
        for (size_t index = 0; index < checkpoint_catalog.size(); ++index) {
            if (index != 0) {
                std::cout << ", ";
            }
            std::cout << "\"" << checkpoint_catalog[index] << "\"";
        }
        std::cout << "]";
    }
    std::cout << "\n}\n";

    llama_batch_free(batch);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
