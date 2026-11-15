#include "qairt-lowering.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <QnnOpDef.h>

namespace ggml::qairt {

bool float_layout(const ggml_tensor * t, bool allow_transpose) noexcept {
    if (!t || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) || t->ne[2] != 1 || t->ne[3] != 1) {
        return false;
    }
    const size_t element = t->type == GGML_TYPE_F32 ? 4 : 2;
    if (t->ne[0] <= 0 || t->ne[1] <= 0 || uint64_t(t->ne[0]) > UINT32_MAX / element ||
        uint64_t(t->ne[1]) > UINT32_MAX / (element * uint64_t(t->ne[0]))) {
        return false;
    }
    const bool dense = t->nb[0] == element && (t->ne[1] == 1 || t->nb[1] == element * size_t(t->ne[0]));
    const bool transposed = t->nb[1] == element && (t->ne[0] == 1 || t->nb[0] == element * size_t(t->ne[1]));
    return dense || (allow_transpose && transposed);
}

size_t dense_bytes(const ggml_tensor * t) noexcept {
    return float_layout(t) ? size_t(t->ne[0]) * size_t(t->ne[1]) * (t->type == GGML_TYPE_F32 ? 4 : 2) : 0;
}

bool metadata_op(enum ggml_op op) noexcept {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW ||
           op == GGML_OP_TRANSPOSE || op == GGML_OP_PERMUTE;
}

const char * elementwise_type(enum ggml_op op) noexcept {
    switch (op) {
        case GGML_OP_ADD: return QNN_OP_ELEMENT_WISE_ADD;
        case GGML_OP_MUL: return QNN_OP_ELEMENT_WISE_MULTIPLY;
        default: return nullptr;
    }
}

bool primitive_supported(const ggml_tensor * t) noexcept {
    if (!float_layout(t, metadata_op(t ? t->op : GGML_OP_NONE))) { return false; }
    if (metadata_op(t->op)) {
        return t->op == GGML_OP_NONE || (t->view_src && float_layout(t->view_src));
    }
    const auto * a = t->src[0];
    if (!float_layout(a)) { return false; }
    if (dense_bytes(t) > 256 * 1024 * 1024 / 4 || dense_bytes(a) > 256 * 1024 * 1024 / 4) { return false; }
    if (t->op == GGML_OP_DUP || t->op == GGML_OP_CONT || t->op == GGML_OP_CPY) {
        return size_t(t->ne[0]) * size_t(t->ne[1]) == size_t(a->ne[0]) * size_t(a->ne[1]);
    }
    constexpr size_t budget_limit = 256 * 1024 * 1024;
    size_t budget = dense_bytes(t) * 3;
    const ggml_tensor * sources[] = {t->src[0], t->src[1]};
    for (const auto * source : sources) {
        if (!float_layout(source) || dense_bytes(source) > budget_limit / 4) { return false; }
        const size_t bytes = dense_bytes(source) * 4;
        if (bytes > budget_limit || budget > budget_limit - bytes) { return false; }
        budget += bytes;
    }
    const auto * b = t->src[1];
    if (!float_layout(b)) { return false; }
    if (t->op == GGML_OP_MUL_MAT) {
        int32_t precision = 0;
        std::memcpy(&precision, t->op_params, sizeof(precision));
        if (precision != GGML_PREC_DEFAULT) { return false; }
        return a->ne[0] == b->ne[0] && t->ne[0] == a->ne[1] && t->ne[1] == b->ne[1] &&
               (a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16) && (b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16);
    }
    return elementwise_type(t->op) && t->type == GGML_TYPE_F16 && a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16 && t->ne[0] == a->ne[0] && t->ne[1] == a->ne[1] &&
           (b->ne[0] == a->ne[0] || b->ne[0] == 1) && (b->ne[1] == a->ne[1] || b->ne[1] == 1);
}

}
