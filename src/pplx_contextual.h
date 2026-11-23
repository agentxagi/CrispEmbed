#pragma once
#include "crispembed.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pplx_contextual_ctx pplx_contextual_ctx;

CRISPEMBED_API pplx_contextual_ctx * pplx_contextual_load(const char * model_path, int n_threads);
CRISPEMBED_API void pplx_contextual_free(pplx_contextual_ctx * ctx);
CRISPEMBED_API int pplx_contextual_dim(const pplx_contextual_ctx * ctx);

// Encodes either a query (is_query=1, n_spans=1, span=[0, n_tokens)) or a multi-chunk
// document (is_query=0, n_spans>=1, span_starts/span_ends in [0, n_tokens]) in one
// bidirectional Qwen3.5 + Gated DeltaNet forward pass. Writes n_spans * dim floats to out_vecs.
// Returns n_spans on success, 0 on error.
CRISPEMBED_API int pplx_contextual_encode_tokens(
    pplx_contextual_ctx * ctx,
    const int32_t * token_ids,
    int n_tokens,
    const int32_t * span_starts,
    const int32_t * span_ends,
    int n_spans,
    int is_query,
    int normalize,
    float * out_vecs);

#ifdef __cplusplus
}
#endif
