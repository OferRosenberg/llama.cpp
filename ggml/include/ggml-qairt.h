#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_backend_qairt_stats {
    uint64_t preparations;
    uint64_t cache_hits;
    uint64_t executions;
    uint64_t w4_preparations;
    uint64_t w4_executions;
    uint64_t preparation_time_us;
    uint64_t finalization_time_us;
    uint64_t finalization_failures;
    uint64_t exported_tensors;
    uint64_t fp16_handoffs;
    uint64_t materialized_outputs;
};

typedef bool (*ggml_backend_qairt_get_stats_t)(ggml_backend_t backend, struct ggml_backend_qairt_stats * stats);

GGML_BACKEND_API ggml_backend_t ggml_backend_qairt_init(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_qairt_reg(void);

#ifdef __cplusplus
}
#endif
