// GPU-only GGML FlashAttention authority adapter for the Llama K0 diagnostic.
//
// This is intentionally dynamically bound: Hawking's normal runtime neither
// links nor depends on a local GGML installation. The Rust caller gates this
// bridge behind HAWKING_LLAMA_GGML_FATTN_AUTHORITY=1 and never treats it as a
// performance path or a promotable K0 result. Its sole purpose is to answer
// whether the installed Metal authority primitive closes the remaining
// one-ULP custom-attention residual.

#include <dlfcn.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ggml_context ggml_context;
typedef struct ggml_tensor ggml_tensor;
typedef struct ggml_cgraph ggml_cgraph;
typedef struct ggml_backend ggml_backend;
typedef struct ggml_backend_buffer ggml_backend_buffer;
typedef struct ggml_backend_buffer_type ggml_backend_buffer_type;
typedef struct ggml_gallocr ggml_gallocr;
typedef struct ggml_backend_reg ggml_backend_reg;
typedef struct ggml_backend_device ggml_backend_device;

typedef ggml_backend * ggml_backend_t;
typedef ggml_backend_buffer * ggml_backend_buffer_t;
typedef ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef ggml_gallocr * ggml_gallocr_t;
typedef ggml_backend_reg * ggml_backend_reg_t;
typedef ggml_backend_device * ggml_backend_dev_t;

typedef struct {
    size_t mem_size;
    void * mem_buffer;
    bool no_alloc;
} ggml_init_params;

enum {
    HAWKING_GGML_TYPE_F32 = 0,
    HAWKING_GGML_TYPE_F16 = 1,
    HAWKING_GGML_PREC_F32 = 10,
    HAWKING_GGML_STATUS_SUCCESS = 0,
    HAWKING_HEAD_DIM = 128,
    HAWKING_CACHE_CAPACITY = 256,
};

typedef struct {
    ggml_context * (*init)(ggml_init_params params);
    void (*free_context)(ggml_context * context);
    ggml_tensor * (*new_tensor_4d)(
        ggml_context * context, int type, int64_t ne0, int64_t ne1,
        int64_t ne2, int64_t ne3);
    ggml_tensor * (*view_4d)(
        ggml_context * context, ggml_tensor * source, int64_t ne0,
        int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2,
        size_t nb3, size_t offset);
    ggml_tensor * (*flash_attn_ext)(
        ggml_context * context, ggml_tensor * q, ggml_tensor * k,
        ggml_tensor * v, ggml_tensor * mask, float scale, float max_bias,
        float logit_softcap);
    void (*flash_attn_ext_set_prec)(ggml_tensor * tensor, int precision);
    ggml_tensor * (*mul_mat)(
        ggml_context * context, ggml_tensor * weights, ggml_tensor * input);
    ggml_cgraph * (*new_graph_custom)(ggml_context * context, size_t size, bool grads);
    void (*build_forward_expand)(ggml_cgraph * graph, ggml_tensor * tensor);
    ggml_backend_reg_t (*backend_reg_by_name)(const char * name);
    ggml_backend_reg_t (*backend_load)(const char * path);
    size_t (*backend_reg_dev_count)(ggml_backend_reg_t registry);
    ggml_backend_dev_t (*backend_reg_dev_get)(ggml_backend_reg_t registry, size_t index);
    ggml_backend_t (*backend_dev_init)(ggml_backend_dev_t device, const char * params);
    void (*backend_free)(ggml_backend_t backend);
    ggml_backend_buffer_t (*backend_alloc_ctx_tensors)(ggml_context * context, ggml_backend_t backend);
    void (*backend_buffer_free)(ggml_backend_buffer_t buffer);
    void (*backend_tensor_set)(ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void (*backend_tensor_get)(const ggml_tensor * tensor, void * data, size_t offset, size_t size);
    ggml_backend_buffer_type_t (*backend_get_default_buffer_type)(ggml_backend_t backend);
    ggml_gallocr_t (*gallocr_new)(ggml_backend_buffer_type_t buffer_type);
    void (*gallocr_free)(ggml_gallocr_t allocator);
    bool (*gallocr_alloc_graph)(ggml_gallocr_t allocator, ggml_cgraph * graph);
    int (*backend_graph_compute)(ggml_backend_t backend, ggml_cgraph * graph);
} hawking_ggml_api;

static void set_error(char * error, size_t error_len, const char * message) {
    if (error != NULL && error_len != 0) {
        (void) snprintf(error, error_len, "%s", message);
    }
}

static const char * env_or_default(const char * name, const char * fallback) {
    const char * value = getenv(name);
    return value != NULL && value[0] != '\0' ? value : fallback;
}

static bool load_api(hawking_ggml_api * api, char * error, size_t error_len) {
    static void * base_handle = NULL;
    static void * registry_handle = NULL;
    if (base_handle == NULL) {
        base_handle = dlopen(
            env_or_default(
                "HAWKING_GGML_BASE_LIBRARY",
                "/opt/homebrew/opt/ggml/lib/libggml-base.dylib"),
            RTLD_NOW | RTLD_GLOBAL);
        if (base_handle == NULL) {
            set_error(error, error_len, "could not load libggml-base.dylib");
            return false;
        }
    }
    if (registry_handle == NULL) {
        registry_handle = dlopen(
            env_or_default(
                "HAWKING_GGML_REGISTRY_LIBRARY",
                "/opt/homebrew/opt/ggml/lib/libggml.dylib"),
            RTLD_NOW | RTLD_GLOBAL);
        if (registry_handle == NULL) {
            set_error(error, error_len, "could not load libggml.dylib");
            return false;
        }
    }
#define HAWKING_LOAD(symbol) \
    do { \
        *(void **) (&api->symbol) = dlsym(RTLD_DEFAULT, "ggml_" #symbol); \
        if (api->symbol == NULL) { \
            set_error(error, error_len, "GGML authority ABI symbol missing: ggml_" #symbol); \
            return false; \
        } \
    } while (false)
    HAWKING_LOAD(init);
    *(void **) (&api->free_context) = dlsym(RTLD_DEFAULT, "ggml_free");
    if (api->free_context == NULL) {
        set_error(error, error_len, "GGML authority ABI symbol missing: ggml_free");
        return false;
    }
    HAWKING_LOAD(new_tensor_4d);
    HAWKING_LOAD(view_4d);
    HAWKING_LOAD(flash_attn_ext);
    HAWKING_LOAD(flash_attn_ext_set_prec);
    HAWKING_LOAD(mul_mat);
    HAWKING_LOAD(new_graph_custom);
    HAWKING_LOAD(build_forward_expand);
    HAWKING_LOAD(backend_reg_by_name);
    HAWKING_LOAD(backend_load);
    HAWKING_LOAD(backend_reg_dev_count);
    HAWKING_LOAD(backend_reg_dev_get);
    HAWKING_LOAD(backend_dev_init);
    HAWKING_LOAD(backend_free);
    HAWKING_LOAD(backend_alloc_ctx_tensors);
    HAWKING_LOAD(backend_buffer_free);
    HAWKING_LOAD(backend_tensor_set);
    HAWKING_LOAD(backend_tensor_get);
    HAWKING_LOAD(backend_get_default_buffer_type);
    HAWKING_LOAD(gallocr_new);
    HAWKING_LOAD(gallocr_free);
    HAWKING_LOAD(gallocr_alloc_graph);
    HAWKING_LOAD(backend_graph_compute);
#undef HAWKING_LOAD
    return true;
}

int hawking_ggml_fattn_f16_authority(
        const float * q,
        const uint16_t * k_active,
        const uint16_t * v_active,
        uint32_t seq_len,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        float * output,
        char * error,
        size_t error_len) {
    if (q == NULL || k_active == NULL || v_active == NULL || output == NULL ||
        seq_len == 0 || seq_len > HAWKING_CACHE_CAPACITY || n_heads == 0 ||
        n_kv_heads == 0 || n_heads % n_kv_heads != 0) {
        set_error(error, error_len, "invalid GGML FATTN authority adapter shape");
        return 1;
    }
    hawking_ggml_api api = {0};
    if (!load_api(&api, error, error_len)) {
        return 1;
    }
    // v0.13.1 registers this plugin as "MTL". Probe the descriptive legacy
    // name too so the diagnostic remains compatible with nearby builds, but
    // prefer the concrete registry name to avoid loading one plugin instance
    // per transformer layer.
    ggml_backend_reg_t registry = api.backend_reg_by_name("MTL");
    if (registry == NULL) {
        registry = api.backend_reg_by_name("Metal");
    }
    if (registry == NULL) {
        registry = api.backend_load(env_or_default(
            "HAWKING_GGML_METAL_PLUGIN",
            "/opt/homebrew/Cellar/ggml/0.13.1/libexec/libggml-metal.so"));
    }
    if (registry == NULL || api.backend_reg_dev_count(registry) == 0) {
        set_error(error, error_len, "could not load GGML Metal authority plugin");
        return 1;
    }

    int result_code = 1;
    ggml_backend_t backend = api.backend_dev_init(api.backend_reg_dev_get(registry, 0), NULL);
    ggml_context * tensor_context = NULL;
    ggml_context * graph_context = NULL;
    ggml_backend_buffer_t input_buffer = NULL;
    ggml_gallocr_t graph_allocator = NULL;
    uint16_t * k_full = NULL;
    uint16_t * v_full = NULL;
    uint16_t mask[HAWKING_CACHE_CAPACITY];
    if (backend == NULL) {
        set_error(error, error_len, "could not initialize GGML Metal authority backend");
        goto cleanup;
    }

    const size_t cache_elements =
        (size_t) HAWKING_HEAD_DIM * HAWKING_CACHE_CAPACITY * n_kv_heads;
    k_full = calloc(cache_elements, sizeof(*k_full));
    v_full = calloc(cache_elements, sizeof(*v_full));
    if (k_full == NULL || v_full == NULL) {
        set_error(error, error_len, "could not allocate authority FATTN cache staging");
        goto cleanup;
    }
    const size_t active_elements =
        (size_t) seq_len * n_kv_heads * HAWKING_HEAD_DIM;
    // Hawking's active cache is already position-major [token, kv_head, D].
    // This is the backing order used by llama.cpp's permuted FATTN views, so
    // preserve it byte-for-byte instead of making a head-major staging copy.
    memcpy(k_full, k_active, active_elements * sizeof(*k_full));
    memcpy(v_full, v_active, active_elements * sizeof(*v_full));
    for (uint32_t position = 0; position < HAWKING_CACHE_CAPACITY; ++position) {
        mask[position] = position < seq_len ? UINT16_C(0x0000) : UINT16_C(0xfc00);
    }

    const ggml_init_params tensor_params = {
        // Tensor object metadata only (`no_alloc = true`). The installed
        // ggml v0.13.1 reports 276-byte tensor objects on this build, so a
        // 4 KiB pool safely holds the four FATTN inputs without baking that
        // private struct size into the adapter ABI.
        .mem_size = 4096,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    tensor_context = api.init(tensor_params);
    if (tensor_context == NULL) {
        set_error(error, error_len, "could not allocate GGML authority tensor context");
        goto cleanup;
    }
    ggml_tensor * q_tensor = api.new_tensor_4d(
        tensor_context, HAWKING_GGML_TYPE_F32, HAWKING_HEAD_DIM, 1, n_heads, 1);
    // llama.cpp stores cache tokens as [D, kv_head, cache_position], then
    // hands FATTN a permuted [D, cache_position, kv_head] view. Preserve
    // that logical shape *and* non-contiguous stride exactly. In particular,
    // the active cache is still 256 logical positions with the invalid tail
    // encoded in the F16 mask; shortening that dimension changes Metal's
    // reduction schedule.
    ggml_tensor * k_storage = api.new_tensor_4d(
        tensor_context, HAWKING_GGML_TYPE_F16, HAWKING_HEAD_DIM, n_kv_heads, HAWKING_CACHE_CAPACITY, 1);
    ggml_tensor * v_storage = api.new_tensor_4d(
        tensor_context, HAWKING_GGML_TYPE_F16, HAWKING_HEAD_DIM, n_kv_heads, HAWKING_CACHE_CAPACITY, 1);
    ggml_tensor * mask_tensor = api.new_tensor_4d(
        tensor_context, HAWKING_GGML_TYPE_F16, HAWKING_CACHE_CAPACITY, 1, 1, 1);
    const size_t row_stride = (size_t) HAWKING_HEAD_DIM * sizeof(*k_full);
    const size_t cache_position_stride = row_stride * n_kv_heads;
    const size_t cache_batch_stride = cache_position_stride * HAWKING_CACHE_CAPACITY;
    ggml_tensor * k_tensor = k_storage == NULL ? NULL : api.view_4d(
        tensor_context, k_storage, HAWKING_HEAD_DIM, HAWKING_CACHE_CAPACITY, n_kv_heads, 1,
        cache_position_stride, row_stride, cache_batch_stride, 0);
    ggml_tensor * v_tensor = v_storage == NULL ? NULL : api.view_4d(
        tensor_context, v_storage, HAWKING_HEAD_DIM, HAWKING_CACHE_CAPACITY, n_kv_heads, 1,
        cache_position_stride, row_stride, cache_batch_stride, 0);
    if (q_tensor == NULL || k_storage == NULL || v_storage == NULL ||
        k_tensor == NULL || v_tensor == NULL || mask_tensor == NULL) {
        set_error(error, error_len, "could not create GGML authority tensors");
        goto cleanup;
    }
    input_buffer = api.backend_alloc_ctx_tensors(tensor_context, backend);
    if (input_buffer == NULL) {
        set_error(error, error_len, "could not allocate GGML authority input buffer");
        goto cleanup;
    }
    api.backend_tensor_set(q_tensor, q, 0, (size_t) n_heads * HAWKING_HEAD_DIM * sizeof(*q));
    api.backend_tensor_set(k_storage, k_full, 0, cache_elements * sizeof(*k_full));
    api.backend_tensor_set(v_storage, v_full, 0, cache_elements * sizeof(*v_full));
    api.backend_tensor_set(mask_tensor, mask, 0, sizeof(mask));

    const ggml_init_params graph_params = {
        .mem_size = 2 * 256 + 16384,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    graph_context = api.init(graph_params);
    if (graph_context == NULL) {
        set_error(error, error_len, "could not allocate GGML authority graph context");
        goto cleanup;
    }
    ggml_tensor * result = api.flash_attn_ext(
        graph_context, q_tensor, k_tensor, v_tensor, mask_tensor,
        1.0f / sqrtf((float) HAWKING_HEAD_DIM), 0.0f, 0.0f);
    ggml_cgraph * graph = api.new_graph_custom(graph_context, 16, false);
    if (result == NULL || graph == NULL) {
        set_error(error, error_len, "could not build GGML authority FATTN graph");
        goto cleanup;
    }
    // llama.cpp b9430 marks this FATTN node GGML_PREC_F32. The default
    // precision selects a different Metal kernel, even with equal inputs.
    api.flash_attn_ext_set_prec(result, HAWKING_GGML_PREC_F32);
    api.build_forward_expand(graph, result);
    graph_allocator = api.gallocr_new(api.backend_get_default_buffer_type(backend));
    if (graph_allocator == NULL || !api.gallocr_alloc_graph(graph_allocator, graph)) {
        set_error(error, error_len, "could not allocate GGML authority FATTN graph");
        goto cleanup;
    }
    if (api.backend_graph_compute(backend, graph) != HAWKING_GGML_STATUS_SUCCESS) {
        set_error(error, error_len, "GGML authority FATTN compute failed");
        goto cleanup;
    }
    api.backend_tensor_get(
        result, output, 0, (size_t) n_heads * HAWKING_HEAD_DIM * sizeof(*output));
    result_code = 0;

cleanup:
    if (graph_allocator != NULL) {
        api.gallocr_free(graph_allocator);
    }
    if (graph_context != NULL) {
        api.free_context(graph_context);
    }
    if (input_buffer != NULL) {
        api.backend_buffer_free(input_buffer);
    }
    if (tensor_context != NULL) {
        api.free_context(tensor_context);
    }
    if (backend != NULL) {
        api.backend_free(backend);
    }
    free(k_full);
    free(v_full);
    return result_code;
}

int hawking_ggml_f16_matvec_authority(
        const uint16_t * weights,
        uint32_t rows,
        uint32_t cols,
        const float * input,
        float * output,
        char * error,
        size_t error_len) {
    if (weights == NULL || input == NULL || output == NULL || rows == 0 || cols == 0) {
        set_error(error, error_len, "invalid GGML f16 matvec authority adapter shape");
        return 1;
    }
    hawking_ggml_api api = {0};
    if (!load_api(&api, error, error_len)) {
        return 1;
    }
    ggml_backend_reg_t registry = api.backend_reg_by_name("MTL");
    if (registry == NULL) {
        registry = api.backend_reg_by_name("Metal");
    }
    if (registry == NULL) {
        registry = api.backend_load(env_or_default(
            "HAWKING_GGML_METAL_PLUGIN",
            "/opt/homebrew/Cellar/ggml/0.13.1/libexec/libggml-metal.so"));
    }
    if (registry == NULL || api.backend_reg_dev_count(registry) == 0) {
        set_error(error, error_len, "could not load GGML Metal authority plugin");
        return 1;
    }

    int result_code = 1;
    ggml_backend_t backend = api.backend_dev_init(api.backend_reg_dev_get(registry, 0), NULL);
    ggml_context * tensor_context = NULL;
    ggml_context * graph_context = NULL;
    ggml_backend_buffer_t input_buffer = NULL;
    ggml_gallocr_t graph_allocator = NULL;
    if (backend == NULL) {
        set_error(error, error_len, "could not initialize GGML Metal authority backend");
        goto cleanup;
    }
    const ggml_init_params tensor_params = {
        .mem_size = 4096,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    tensor_context = api.init(tensor_params);
    if (tensor_context == NULL) {
        set_error(error, error_len, "could not allocate GGML authority matvec tensor context");
        goto cleanup;
    }
    // GGML stores matrices as [input_cols, output_rows]. The incoming
    // Hawking f16 table is row-major, so its physical order already matches
    // this transposed logical shape without a repack.
    ggml_tensor * weight_tensor = api.new_tensor_4d(
        tensor_context, HAWKING_GGML_TYPE_F16, cols, rows, 1, 1);
    ggml_tensor * input_tensor = api.new_tensor_4d(
        tensor_context, HAWKING_GGML_TYPE_F32, cols, 1, 1, 1);
    if (weight_tensor == NULL || input_tensor == NULL) {
        set_error(error, error_len, "could not create GGML authority matvec tensors");
        goto cleanup;
    }
    input_buffer = api.backend_alloc_ctx_tensors(tensor_context, backend);
    if (input_buffer == NULL) {
        set_error(error, error_len, "could not allocate GGML authority matvec input buffer");
        goto cleanup;
    }
    const size_t weight_bytes = (size_t) rows * cols * sizeof(*weights);
    api.backend_tensor_set(weight_tensor, weights, 0, weight_bytes);
    api.backend_tensor_set(input_tensor, input, 0, (size_t) cols * sizeof(*input));

    const ggml_init_params graph_params = {
        .mem_size = 2 * 256 + 16384,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    graph_context = api.init(graph_params);
    if (graph_context == NULL) {
        set_error(error, error_len, "could not allocate GGML authority matvec graph context");
        goto cleanup;
    }
    ggml_tensor * result = api.mul_mat(graph_context, weight_tensor, input_tensor);
    ggml_cgraph * graph = api.new_graph_custom(graph_context, 16, false);
    if (result == NULL || graph == NULL) {
        set_error(error, error_len, "could not build GGML authority matvec graph");
        goto cleanup;
    }
    api.build_forward_expand(graph, result);
    graph_allocator = api.gallocr_new(api.backend_get_default_buffer_type(backend));
    if (graph_allocator == NULL || !api.gallocr_alloc_graph(graph_allocator, graph)) {
        set_error(error, error_len, "could not allocate GGML authority matvec graph");
        goto cleanup;
    }
    if (api.backend_graph_compute(backend, graph) != HAWKING_GGML_STATUS_SUCCESS) {
        set_error(error, error_len, "GGML authority matvec compute failed");
        goto cleanup;
    }
    api.backend_tensor_get(result, output, 0, (size_t) rows * sizeof(*output));
    result_code = 0;

cleanup:
    if (graph_allocator != NULL) {
        api.gallocr_free(graph_allocator);
    }
    if (graph_context != NULL) {
        api.free_context(graph_context);
    }
    if (input_buffer != NULL) {
        api.backend_buffer_free(input_buffer);
    }
    if (tensor_context != NULL) {
        api.free_context(tensor_context);
    }
    if (backend != NULL) {
        api.backend_free(backend);
    }
    return result_code;
}
