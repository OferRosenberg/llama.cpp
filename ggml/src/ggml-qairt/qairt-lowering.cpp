#include "qairt-lowering.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <QnnOpDef.h>

namespace ggml::qairt {

static bool matrix_layout(const ggml_tensor * t, bool allow_transpose) noexcept {
    if (!t || t->ne[2] != 1 || t->ne[3] != 1) {
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

bool float_layout(const ggml_tensor * t, bool allow_transpose) noexcept {
    return t && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) && matrix_layout(t, allow_transpose);
}
bool metadata_layout(const ggml_tensor * t) noexcept {
    if (!t || (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_BF16)) { return false; }
    constexpr size_t limit = 256 * 1024 * 1024;
    const size_t element = t->type == GGML_TYPE_F32 ? 4 : 2;
    size_t count = 1, span = element;
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] <= 0 || uint64_t(t->ne[i]) > limit / element / count) { return false; }
        count *= size_t(t->ne[i]);
        const size_t extra = size_t(t->ne[i]) - 1;
        if (t->nb[i] && extra > (limit - span) / t->nb[i]) { return false; }
        span += extra * t->nb[i];
    }
    return true;
}
bool row_layout(const ggml_tensor * t) noexcept {
    return t && (t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32) && metadata_layout(t) && ggml_is_contiguous(t);
}

bool silu_op(const ggml_tensor * t) noexcept {
    return t && t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_SILU;
}

bool index_layout(const ggml_tensor * t) noexcept {
    return t && (t->type == GGML_TYPE_I32 || t->type == GGML_TYPE_I64) && t->ne[0] > 0 && t->ne[0] <= 4096 &&
        t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t);
}

bool position_layout(const ggml_tensor * t) noexcept {
    return t && t->type == GGML_TYPE_I32 && t->ne[0] > 0 && t->ne[0] <= 4096 &&
           t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t);
}
rope_parameters rope_params(const ggml_tensor * t) noexcept {
    int32_t words[16]; std::memcpy(words, t->op_params, sizeof(words));
    rope_parameters p {};
    p.dimensions = words[1]; p.mode = words[2]; p.context = words[4];
    std::memcpy(&p.base, words + 5, sizeof(float));
    std::memcpy(&p.scale, words + 6, sizeof(float));
    std::memcpy(&p.extension, words + 7, sizeof(float));
    std::memcpy(&p.attention, words + 8, sizeof(float));
    std::memcpy(&p.beta_fast, words + 9, sizeof(float));
    std::memcpy(&p.beta_slow, words + 10, sizeof(float));
    p.truncate = ggml_rope_get_truncate(t);
    return p;
}

uint32_t row_height(uint32_t rows) noexcept {
    uint32_t height = rows > 4096 ? std::min(rows, uint32_t(256)) : 1;
    while (rows % height != 0) { --height; }
    return height;
}
const char * rowwise_type(enum ggml_op op) noexcept {
    switch (op) {
        case GGML_OP_RMS_NORM: return QNN_OP_RMS_NORM;
        case GGML_OP_NORM: return QNN_OP_LAYER_NORM;
        case GGML_OP_SOFT_MAX: return QNN_OP_SOFTMAX;
        default: return nullptr;
    }
}
bool weight_layout(const ggml_tensor * t) noexcept {
    if (float_layout(t) || (t && t->type == GGML_TYPE_BF16 && matrix_layout(t, true))) { return true; }
    if (!t || (t->type != GGML_TYPE_Q4_K && t->type != GGML_TYPE_Q6_K) ||
        t->ne[0] <= 0 || t->ne[1] <= 0 || t->ne[2] != 1 || t->ne[3] != 1 ||
        uint64_t(t->ne[0]) > UINT32_MAX / 2 || uint64_t(t->ne[1]) > UINT32_MAX / (2 * uint64_t(t->ne[0]))) { return false; }
    const size_t block = ggml_blck_size(t->type);
    return size_t(t->ne[0]) % block == 0 && t->nb[0] == ggml_type_size(t->type) &&
           t->nb[1] == size_t(t->ne[0]) / block * ggml_type_size(t->type);
}
size_t dense_bytes(const ggml_tensor * t) noexcept {
    return index_layout(t) ? size_t(t->ne[0]) * ggml_type_size(t->type) : weight_layout(t) || metadata_layout(t) ? size_t(ggml_nelements(t)) * (t->type == GGML_TYPE_F32 ? 4 : 2) : 0;
}

bool metadata_op(enum ggml_op op) noexcept {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW ||
           op == GGML_OP_TRANSPOSE || op == GGML_OP_PERMUTE;
}

const char * elementwise_type(enum ggml_op op) noexcept {
    switch (op) {
        case GGML_OP_MUL: return QNN_OP_ELEMENT_WISE_MULTIPLY;
        case GGML_OP_ADD: return QNN_OP_ELEMENT_WISE_ADD;
        default: return nullptr;
    }
}

bool attention_supported(const ggml_tensor * t) noexcept {
    if (!t || t->op != GGML_OP_FLASH_ATTN_EXT || t->type != GGML_TYPE_F32 || !row_layout(t) ||
        ggml_prec_get_acc(t) != GGML_PREC_DEFAULT || ggml_prec_get_src(t, 1) != GGML_PREC_DEFAULT) { return false; }
    const auto * q = t->src[0]; const auto * k = t->src[1]; const auto * v = t->src[2]; const auto * mask = t->src[3];
    if (!q || !k || !v || !mask || !metadata_layout(q) || !metadata_layout(k) || !metadata_layout(v) || !row_layout(mask) ||
        (q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_F16) || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || q->ne[0] != k->ne[0] || k->ne[0] != v->ne[0] ||
        k->ne[1] != v->ne[1] || k->ne[2] != v->ne[2] || q->ne[2] % k->ne[2] || q->ne[1] > k->ne[1] ||
        q->ne[0] > 256 || q->ne[0] % 32 || q->ne[1] > 128 || k->ne[1] > 4096 || q->ne[2] > 64 ||
        mask->ne[0] != k->ne[1] || mask->ne[1] < q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1 ||
        t->ne[0] != v->ne[0] || t->ne[1] != q->ne[2] || t->ne[2] != q->ne[1] || t->ne[3] != 1) { return false; }
    for (size_t i = 4; i < GGML_MAX_SRC; ++i) { if (t->src[i]) { return false; } }
    float params[3]; std::memcpy(params, t->op_params, sizeof(params));
    if (!std::isfinite(params[0]) || params[0] <= 0.0f || params[1] != 0.0f || params[2] != 0.0f) { return false; }
    int32_t sparse; std::memcpy(&sparse, reinterpret_cast<const uint8_t *>(t->op_params) + 4 * sizeof(float), sizeof(sparse));
    if (sparse != 0) { return false; }
    constexpr size_t limit = 256 * 1024 * 1024;
    size_t budget = 0;
    for (const auto * source : {t, q, k, v, mask}) {
        const size_t bytes = size_t(ggml_nelements(source)) * (source->type == GGML_TYPE_F32 ? 4 : 2);
        if (bytes > limit / 12 || budget > limit - bytes * 12) { return false; }
        budget += bytes * 12;
    }
    return true;
}

bool primitive_supported(const ggml_tensor * t) noexcept {
    if (t && t->op == GGML_OP_FLASH_ATTN_EXT) { return attention_supported(t); }
    if (!t) { return false; }
    if (t->op == GGML_OP_NONE && (weight_layout(t) || index_layout(t))) { return true; }
    if (metadata_op(t->op)) {
        if (!metadata_layout(t)) { return false; }
        if (t->op == GGML_OP_NONE) { return true; }
        return t->view_src && metadata_layout(t->view_src) && t->view_offs <= ggml_nbytes(t->view_src) &&
               ggml_nbytes(t) <= ggml_nbytes(t->view_src) - t->view_offs;
    }
    if (t->op == GGML_OP_SET_ROWS) {
        const auto * updates = t->src[0]; const auto * indices = t->src[1]; const auto * cache = t->src[2];
        if (!float_layout(updates, false) || !float_layout(cache, false) || cache->type != GGML_TYPE_F16 || t->type != GGML_TYPE_F16 ||
            !row_layout(t) || !ggml_are_same_shape(t, cache) || !index_layout(indices) ||
            updates->ne[0] != cache->ne[0] || updates->ne[1] != indices->ne[0] || cache->ne[1] > 4096 ||
            dense_bytes(t) > 256 * 1024 * 1024 / 12 || dense_bytes(updates) > 256 * 1024 * 1024 / 12) { return false; }
        for (size_t i = 3; i < GGML_MAX_SRC; ++i) { if (t->src[i]) { return false; } }
        return t->view_src == cache || (t->view_src == cache->view_src && t->view_offs == cache->view_offs);
    }
    if (t->op == GGML_OP_ROPE) {
        const auto * x = t->src[0]; const auto * ids = t->src[1]; const auto * factors = t->src[2];
        if (!row_layout(t) || !row_layout(x) || t->type != x->type || !ggml_are_same_shape(t, x) ||
            !position_layout(ids) || ids->ne[0] != t->ne[2] || t->ne[0] > 4096 || dense_bytes(t) > 256 * 1024 * 1024 / 12) { return false; }
        for (size_t i = 3; i < GGML_MAX_SRC; ++i) { if (t->src[i]) { return false; } }
        const auto p = rope_params(t);
        int32_t offset; std::memcpy(&offset, reinterpret_cast<const uint8_t *>(t->op_params) + 15 * sizeof(int32_t), sizeof(offset));
        if ((p.mode != GGML_ROPE_TYPE_NORMAL && p.mode != GGML_ROPE_TYPE_NEOX) || offset != 0 ||
            p.dimensions < 2 || p.dimensions > t->ne[0] || p.dimensions % 2 || t->ne[0] % 2 ||
            !std::isfinite(p.base) || p.base <= 0.0f || !std::isfinite(p.scale) || p.scale <= 0.0f ||
            !std::isfinite(p.extension) || !std::isfinite(p.attention) ||
            !std::isfinite(p.beta_fast) || !std::isfinite(p.beta_slow)) { return false; }
        if (p.extension != 0.0f && (p.context <= 0 || p.base <= 1.0f || p.beta_fast <= 0.0f || p.beta_slow <= 0.0f)) { return false; }
        return !factors || (float_layout(factors, false) && factors->type == GGML_TYPE_F32 && factors->ne[1] == 1 &&
                            factors->ne[0] >= p.dimensions / 2 && factors->ne[0] <= 4096);
    }
    if (rowwise_type(t->op) || silu_op(t)) {
        const auto * source = t->src[0];
        if (!row_layout(t) || !row_layout(source) || t->type != source->type || !ggml_are_same_shape(t, source)) { return false; }
        for (size_t i = t->op == GGML_OP_SOFT_MAX ? 2 : 1; i < GGML_MAX_SRC; ++i) { if (t->src[i]) { return false; } }
        constexpr size_t limit = 256 * 1024 * 1024;
        if (dense_bytes(t) > limit / 7) { return false; }
        // Large row batches must fit the spatial tiling profile used by qnn_shape.
        const uint32_t rows = uint32_t(ggml_nelements(t) / t->ne[0]);
        if (rows / row_height(rows) > 4096) { return false; }
        if (silu_op(t)) { return true; }
        float parameter; std::memcpy(&parameter, t->op_params, sizeof(parameter));
        if (!std::isfinite(parameter) || parameter <= 0.0f) { return false; }
        if (t->op == GGML_OP_SOFT_MAX) {
            float bias; std::memcpy(&bias, reinterpret_cast<const uint8_t *>(t->op_params) + sizeof(float), sizeof(bias));
            if (bias != 0.0f) { return false; }
            const auto * mask = t->src[1];
            return !mask || (row_layout(mask) && mask->ne[0] == t->ne[0] && mask->ne[1] >= t->ne[1] &&
                            (mask->ne[2] == 1 || mask->ne[2] == t->ne[2]) &&
                            (mask->ne[3] == 1 || mask->ne[3] == t->ne[3]) && dense_bytes(mask) <= limit / 7);
        }
        return true;
    }
    if (t->op == GGML_OP_ADD) {
        const auto * a = t->src[0]; const auto * b = t->src[1];
        if (!row_layout(t) || !row_layout(a) || !row_layout(b) || t->type != a->type || t->type != b->type ||
            !ggml_are_same_shape(t, a)) { return false; }
        for (size_t i = 2; i < GGML_MAX_SRC; ++i) { if (t->src[i]) { return false; } }
        for (size_t i = 0; i < GGML_MAX_DIMS; ++i) { if (b->ne[i] != t->ne[i] && b->ne[i] != 1) { return false; } }
        constexpr size_t limit = 256 * 1024 * 1024;
        size_t budget = 0;
        for (const auto * source : {t, a, b}) {
            const size_t bytes = dense_bytes(source);
            if (bytes > limit / 4 || budget > limit - bytes * 4) { return false; }
            budget += bytes * 4;
        }
        return true;
    }
    if (!float_layout(t, false)) { return false; }
    const auto * a = t->src[0];
    if (!float_layout(a) && !(t->op == GGML_OP_MUL_MAT && weight_layout(a))) { return false; }
    if (dense_bytes(t) > 256 * 1024 * 1024 / 4 || dense_bytes(a) > 256 * 1024 * 1024 / 4) { return false; }
    if (t->op == GGML_OP_DUP || t->op == GGML_OP_CONT || t->op == GGML_OP_CPY) {
        return size_t(t->ne[0]) * size_t(t->ne[1]) == size_t(a->ne[0]) * size_t(a->ne[1]);
    }
    constexpr size_t budget_limit = 256 * 1024 * 1024;
    size_t budget = dense_bytes(t) * 3;
    const ggml_tensor * sources[] = {t->src[0], t->src[1]};
    for (const auto * source : sources) {
        if (!weight_layout(source) || dense_bytes(source) > budget_limit / 4) { return false; }
        const size_t bytes = dense_bytes(source) * 4;
        if (bytes > budget_limit || budget > budget_limit - bytes) { return false; }
        budget += bytes;
    }
    const auto * b = t->src[1];
    if (!float_layout(b) && !(t->op == GGML_OP_MUL_MAT && a->type == GGML_TYPE_BF16 && b->type == GGML_TYPE_BF16 && weight_layout(b))) { return false; }
    if (t->op == GGML_OP_MUL_MAT) {
        if (ggml_prec_get_acc(t) != GGML_PREC_DEFAULT || ggml_prec_get_src(t, 1) != GGML_PREC_DEFAULT) { return false; }
        if (ggml_is_quantized(a->type) && b->type != GGML_TYPE_F32) { return false; }
        if (a->type == GGML_TYPE_BF16 && (t->type != GGML_TYPE_F32 || b->type == GGML_TYPE_F16)) { return false; }
        return a->ne[0] == b->ne[0] && t->ne[0] == a->ne[1] && t->ne[1] == b->ne[1] &&
               weight_layout(a) && (b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16 || b->type == GGML_TYPE_BF16);
    }
    return elementwise_type(t->op) && t->type == GGML_TYPE_F16 && a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16 && t->ne[0] == a->ne[0] && t->ne[1] == a->ne[1] &&
           (b->ne[0] == a->ne[0] || b->ne[0] == 1) && (b->ne[1] == a->ne[1] || b->ne[1] == 1);
}

}
