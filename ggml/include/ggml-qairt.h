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
};

typedef bool (*ggml_backend_qairt_get_stats_t)(ggml_backend_t backend, struct ggml_backend_qairt_stats * stats);

enum ggml_backend_qairt_archive_action {
    GGML_QAIRT_ARCHIVE_BEGIN_PREFILL,
    GGML_QAIRT_ARCHIVE_BEGIN_DECODE,
    GGML_QAIRT_ARCHIVE_PREPARE,
    GGML_QAIRT_ARCHIVE_SAVE,
    GGML_QAIRT_ARCHIVE_LOAD,
    GGML_QAIRT_ARCHIVE_SELECT_PREFILL,
    GGML_QAIRT_ARCHIVE_SELECT_DECODE,
};
// Experimental same-process phase archives, accessed through the backend registry.
typedef bool (*ggml_backend_qairt_archive_t)(ggml_backend_t backend, int action, struct ggml_cgraph * graph, const char * path);

GGML_BACKEND_API ggml_backend_t ggml_backend_qairt_init(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_qairt_reg(void);

#ifdef __cplusplus
}
#endif
