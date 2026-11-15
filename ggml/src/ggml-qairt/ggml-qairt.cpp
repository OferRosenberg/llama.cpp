#include "ggml-qairt.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "qairt-backend.h"

#include <atomic>
#include <cstring>
#include <limits>
#include <malloc.h>
#include <mutex>
#include <new>

using namespace ggml::qairt;

namespace {
std::atomic<uint64_t> next_generation {1};
const char * buffer_type_name(ggml_backend_buffer_type_t) { return "QAIRT"; }

ggml_backend_buffer_t tensor_buffer(const ggml_tensor * t) {
    return t->view_src ? t->view_src->buffer : t->buffer;
}

bool checked_span(const ggml_tensor * t, size_t & bytes) {
    if (!t || t->type < 0 || t->type >= GGML_TYPE_COUNT) { return false; }
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) { if (t->ne[i] <= 0) { return false; } }
    const size_t block = ggml_blck_size(t->type);
    if (!block || size_t(t->ne[0]) % block) { return false; }
    const size_t limit = std::numeric_limits<size_t>::max();
    bytes = ggml_type_size(t->type);
    size_t first = 0;
    if (block != 1) {
        const size_t blocks = size_t(t->ne[0]) / block;
        if (t->nb[0] && blocks > limit / t->nb[0]) { return false; }
        bytes = blocks * t->nb[0]; first = 1;
    }
    for (size_t i = first; i < GGML_MAX_DIMS; ++i) {
        const size_t count = size_t(t->ne[i]) - 1;
        if (t->nb[i] && count > (limit - bytes) / t->nb[i]) { return false; }
        bytes += count * t->nb[i];
    }
    return true;
}

bool checked_access(ggml_backend_buffer_t buffer, const ggml_tensor * t, size_t offset, size_t size) {
    if (!t || !t->data) { return false; }
    const auto * ctx = static_cast<const buffer_context *>(buffer->context);
    const auto base = reinterpret_cast<uintptr_t>(ctx->storage);
    const auto ptr = reinterpret_cast<uintptr_t>(t->data);
    size_t bytes = 0;
    if (!checked_span(t, bytes)) { return false; }
    return ptr >= base && ptr - base <= ctx->size && bytes <= ctx->size - (ptr - base) &&
           offset <= bytes && size <= bytes - offset;
}

void buffer_free(ggml_backend_buffer_t b) {
    auto * c = static_cast<buffer_context *>(b->context);
    _aligned_free(c->storage);
    delete c;
}
void * buffer_base(ggml_backend_buffer_t b) { return static_cast<buffer_context *>(b->context)->storage; }
enum ggml_status buffer_init_tensor(ggml_backend_buffer_t b, ggml_tensor * t) {
    return checked_access(b, t, 0, 0) ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}
void buffer_set(ggml_backend_buffer_t b, ggml_tensor * t, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(checked_access(b, t, offset, size) && (data || !size));
    if (size) { std::memmove(static_cast<uint8_t *>(t->data) + offset, data, size); }
    static_cast<buffer_context *>(b->context)->touch();
}
void buffer_get(ggml_backend_buffer_t b, const ggml_tensor * t, void * data, size_t offset, size_t size) {
    GGML_ASSERT(checked_access(b, t, offset, size) && (data || !size));
    if (size) { std::memmove(data, static_cast<const uint8_t *>(t->data) + offset, size); }
}
void buffer_memset(ggml_backend_buffer_t b, ggml_tensor * t, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(checked_access(b, t, offset, size));
    if (size) { std::memset(static_cast<uint8_t *>(t->data) + offset, value, size); }
    static_cast<buffer_context *>(b->context)->touch();
}
void buffer_set_n(ggml_backend_buffer_t b, ggml_tensor * const * ts, const void * const * data,
                  const size_t * offsets, const size_t * sizes, size_t count) {
    for (size_t i = 0; i < count; ++i) { GGML_ASSERT(checked_access(b, ts[i], offsets[i], sizes[i]) && (data[i] || !sizes[i])); }
    for (size_t i = 0; i < count; ++i) { buffer_set(b, ts[i], data[i], offsets[i], sizes[i]); }
}
void buffer_get_n(ggml_backend_buffer_t b, const ggml_tensor * const * ts, void * const * data,
                  const size_t * offsets, const size_t * sizes, size_t count) {
    for (size_t i = 0; i < count; ++i) { GGML_ASSERT(checked_access(b, ts[i], offsets[i], sizes[i]) && (data[i] || !sizes[i])); }
    for (size_t i = 0; i < count; ++i) { buffer_get(b, ts[i], data[i], offsets[i], sizes[i]); }
}
bool can_copy(ggml_backend_buffer_t b, const ggml_tensor * src, ggml_tensor * dst) {
    size_t bytes = 0;
    if (!checked_span(src, bytes) || !src->data) { return false; }
    auto source = tensor_buffer(src);
    return source && ggml_backend_buft_is_host(source->buft) && src->data &&
           ggml_are_same_layout(src, dst) && checked_access(b, dst, 0, bytes);
}
bool buffer_copy(ggml_backend_buffer_t b, const ggml_tensor * src, ggml_tensor * dst) {
    if (!can_copy(b, src, dst)) { return false; }
    buffer_set(b, dst, src->data, 0, ggml_nbytes(src));
    return true;
}
bool buffer_copy_n(ggml_backend_buffer_t b, const ggml_tensor * const * src, ggml_tensor * const * dst, size_t count) {
    // Cross-pair aliases need a staging transaction; leave them to GGML's normal copy path.
    for (size_t i = 0; i < count; ++i) {
        if (!can_copy(b, src[i], dst[i])) { return false; }
        for (size_t j = 0; j < count; ++j) {
            if (i == j) { continue; }
            const auto a = reinterpret_cast<uintptr_t>(src[i]->data);
            const auto z = reinterpret_cast<uintptr_t>(dst[j]->data);
            if ((a <= z && z - a < ggml_nbytes(src[i])) || (z < a && a - z < ggml_nbytes(dst[j]))) { return false; }
        }
    }
    for (size_t i = 0; i < count; ++i) { buffer_copy(b, src[i], dst[i]); }
    return true;
}
void buffer_clear(ggml_backend_buffer_t b, uint8_t value) {
    auto * c = static_cast<buffer_context *>(b->context);
    std::memset(c->storage, value, c->size);
    c->touch();
}
void buffer_reset(ggml_backend_buffer_t b) { static_cast<buffer_context *>(b->context)->touch(); }

ggml_backend_buffer_t buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    try {
        if (size > std::numeric_limits<size_t>::max() - 63) { return nullptr; }
        auto c = std::make_unique<buffer_context>();
        c->size = size;
        c->identity = next_generation.fetch_add(1);
        c->touch();
        c->storage = _aligned_malloc(size ? (size + 63) & ~size_t(63) : 64, 64);
        if (!c->storage) { return nullptr; }
        ggml_backend_buffer_i iface {};
        iface.free_buffer = buffer_free; iface.get_base = buffer_base; iface.init_tensor = buffer_init_tensor;
        iface.memset_tensor = buffer_memset; iface.set_tensor = buffer_set; iface.get_tensor = buffer_get;
        iface.cpy_tensor = buffer_copy; iface.clear = buffer_clear; iface.reset = buffer_reset;
        iface.set_tensor_n = buffer_set_n; iface.get_tensor_n = buffer_get_n; iface.cpy_tensor_n = buffer_copy_n;
        auto b = ggml_backend_buffer_init(buft, iface, c.get(), size);
        if (!b) { _aligned_free(c->storage); return nullptr; }
        c.release();
        return b;
    } catch (...) { return nullptr; }
}
size_t buffer_alignment(ggml_backend_buffer_type_t) { return 64; }
bool buffer_host(ggml_backend_buffer_type_t) { return true; }

struct registry_context {
    std::weak_ptr<runtime_owner> runtime;
    std::shared_ptr<const runtime_config> config;
    std::mutex mutex;
    bool available = false;
    std::string description;
    ggml_backend_reg reg {};
    ggml_backend_device device {};
    ggml_backend_buffer_type buft {};
};
const char * device_name(ggml_backend_dev_t) { return "QAIRT0"; }
const char * device_description(ggml_backend_dev_t dev) { return static_cast<registry_context *>(dev->context)->description.c_str(); }
void device_memory(ggml_backend_dev_t, size_t * available, size_t * total) { *available = 0; *total = 0; }
enum ggml_backend_dev_type device_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
void device_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    *props = {};
    props->name = device_name(dev); props->description = device_description(dev); props->type = device_type(dev);
}
ggml_backend_buffer_type_t device_buft(ggml_backend_dev_t dev) { return &static_cast<registry_context *>(dev->context)->buft; }
bool device_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return buft && (buft->iface.get_name == buffer_type_name || ggml_backend_buft_is_host(buft));
}
bool device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * t) {
    try {
        if (!primitive_supported(t)) { return false; }
        const ggml_tensor * operands[] = {t, t->src[0], t->src[1]};
        for (const auto * s : operands) {
            if (!s) { continue; }
            auto b = tensor_buffer(s);
            if (b && !device_supports_buft(dev, b->buft)) { return false; }
        }
        return true;
    } catch (...) { return false; }
}
bool device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * t) {
    // Only numerical primitives request offload; layout helpers follow their partition.
    return t && (t->op == GGML_OP_MUL_MAT || elementwise_type(t->op)) && device_supports_op(dev, t);
}
const char * backend_name(ggml_backend_t) { return "QAIRT"; }
void backend_free(ggml_backend_t backend) {
    delete static_cast<backend_context *>(backend->context);
    delete backend;
}
enum ggml_status backend_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    try { return static_cast<backend_context *>(backend->context)->compute(graph); }
    catch (const std::exception & e) { GGML_LOG_ERROR("qairt: graph_compute: %s\n", e.what()); }
    catch (...) { GGML_LOG_ERROR("qairt: graph_compute: unknown exception\n"); }
    static_cast<backend_context *>(backend->context)->invalidate();
    return GGML_STATUS_FAILED;
}
ggml_backend_t device_init(ggml_backend_dev_t dev, const char * params) {
    try {
        if (params && *params) { GGML_LOG_ERROR("qairt: backend parameters are not supported; use environment configuration\n"); return nullptr; }
        auto * state = static_cast<registry_context *>(dev->context);
        std::lock_guard<std::mutex> lock(state->mutex);
        auto runtime = state->runtime.lock();
        if (!runtime) {
            auto created = runtime_factory::create(state->config);
            if (!created.ok()) { GGML_LOG_ERROR("qairt: %s: %s\n", created.error().operation.c_str(), created.error().message.c_str()); return nullptr; }
            runtime = std::move(created).value();
            state->runtime = runtime;
        }
        auto c = std::make_unique<backend_context>(std::move(runtime));
        static ggml_guid guid = {0x71,0xa1,0x52,0x01,0x13,0x08,0x41,0x91,0xb1,0x32,0x67,0x85,0x22,0x10,0xe4,0x90};
        ggml_backend_i iface {};
        iface.get_name = backend_name; iface.free = backend_free; iface.graph_compute = backend_compute;
        auto * backend = new ggml_backend { &guid, iface, dev, c.get() };
        c.release();
        return backend;
    } catch (...) { return nullptr; }
}
const char * registry_name(ggml_backend_reg_t) { return "QAIRT"; }
size_t registry_count(ggml_backend_reg_t reg) { return static_cast<registry_context *>(reg->context)->available ? 1 : 0; }
ggml_backend_dev_t registry_device(ggml_backend_reg_t reg, size_t index) {
    return index == 0 && registry_count(reg) ? &static_cast<registry_context *>(reg->context)->device : nullptr;
}
bool get_stats(ggml_backend_t backend, ggml_backend_qairt_stats * stats) {
    if (!backend || !stats || backend->iface.graph_compute != backend_compute) { return false; }
    *stats = static_cast<backend_context *>(backend->context)->stats();
    return true;
}
void * registry_proc(ggml_backend_reg_t, const char * name) {
    return name && std::strcmp(name, "ggml_backend_qairt_get_stats") == 0 ? reinterpret_cast<void *>(get_stats) : nullptr;
}
}

namespace ggml::qairt {
void buffer_context::touch() noexcept { generation = next_generation.fetch_add(1); }
bool qairt_buffer(ggml_backend_buffer_t b) noexcept { return b && b->buft->iface.get_name == buffer_type_name; }
bool readable_tensor(const ggml_tensor * t) noexcept {
    if (!float_layout(t) || !t->data) { return false; }
    auto b = tensor_buffer(t);
    if (!b || !ggml_backend_buft_is_host(b->buft)) { return false; }
    const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(b));
    const auto ptr = reinterpret_cast<uintptr_t>(t->data);
    const size_t size = ggml_backend_buffer_get_size(b);
    return ptr >= base && ptr - base <= size && ggml_nbytes(t) <= size - (ptr - base);
}
}

extern "C" ggml_backend_reg_t ggml_backend_qairt_reg(void) {
    try {
        static registry_context state;
        static std::once_flag once;
        std::call_once(once, [&] {
            state.reg = { GGML_BACKEND_API_VERSION, {registry_name, registry_count, registry_device, registry_proc}, &state };
            auto config = process_config::capture_once();
            if (!config.ok()) { GGML_LOG_ERROR("qairt: configuration: %s\n", config.error().message.c_str()); return; }
            auto runtime = runtime_factory::create(config.value());
            if (!runtime.ok()) { GGML_LOG_ERROR("qairt: %s: %s\n", runtime.error().operation.c_str(), runtime.error().message.c_str()); return; }
            state.config = config.value();
            auto owner = std::move(runtime).value();
            state.description = owner->policy().descriptor().display_name + " (FP16 primitives, M1)";
            if (owner->platform().htp_architecture) { state.description += " arch " + std::to_string(*owner->platform().htp_architecture); }
            state.device.reg = &state.reg; state.device.context = &state;
            auto & d = state.device.iface;
            d.get_name = device_name; d.get_description = device_description; d.get_memory = device_memory;
            d.get_type = device_type; d.get_props = device_props; d.init_backend = device_init;
            d.get_buffer_type = device_buft; d.supports_op = device_supports_op;
            d.supports_buft = device_supports_buft; d.offload_op = device_offload_op;
            state.buft.device = &state.device; state.buft.context = &state;
            state.buft.iface.get_name = buffer_type_name; state.buft.iface.alloc_buffer = buffer_alloc;
            state.buft.iface.get_alignment = buffer_alignment; state.buft.iface.is_host = buffer_host;
            auto closed = owner->close();
            if (!closed.ok()) { GGML_LOG_ERROR("qairt: discovery cleanup failed\n"); return; }
            state.available = true;
        });
        return &state.reg;
    } catch (...) { GGML_LOG_ERROR("qairt: registry initialization failed\n"); return nullptr; }
}
extern "C" ggml_backend_t ggml_backend_qairt_init(void) {
    auto reg = ggml_backend_qairt_reg();
    auto dev = reg ? registry_device(reg, 0) : nullptr;
    return dev ? device_init(dev, nullptr) : nullptr;
}
GGML_BACKEND_DL_IMPL(ggml_backend_qairt_reg)
