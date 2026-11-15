#pragma once

#include "qairt-target.h"

#include <atomic>
#include <mutex>
#include <type_traits>

namespace ggml::qairt {

struct provider_identity {
    uint32_t backend_id = 0;
    Qnn_ApiVersion_t version {};
    std::string provider_name;
    std::filesystem::path module_path;
};

class dll_module final {
public:
    static result<std::shared_ptr<dll_module>> load(const std::filesystem::path & path);
    ~dll_module() noexcept;
    dll_module(const dll_module &) = delete;
    dll_module & operator=(const dll_module &) = delete;

    template<class T>
    result<T> resolve(const char * symbol) const {
        static_assert(std::is_pointer<T>::value && std::is_function<typename std::remove_pointer<T>::type>::value);
        auto address = resolve_address(symbol);
        if (!address.ok()) { return result<T>::failure(address.error()); }
        return result<T>::success(reinterpret_cast<T>(address.value()));
    }
    const std::filesystem::path & canonical_path() const noexcept { return path_; }

private:
    dll_module();
    using symbol_address = void (*)();
    result<symbol_address> resolve_address(const char * symbol) const;
    struct native_module;
    std::unique_ptr<native_module> native_;
    std::filesystem::path path_;
};

class qnn_api final {
public:
    static result<qnn_api> select_provider(const dll_module & module);
    static result<qnn_api> from_providers(const QnnInterface_t * const * providers, uint32_t count);
    const provider_identity & identity() const noexcept { return identity_; }
    status validate_required_functions(bool profile) const;
private:
    friend class api_domain;
    QNN_INTERFACE_VER_TYPE table_ = QNN_INTERFACE_VER_TYPE_INIT;
    provider_identity identity_;
};

struct cleanup_statistics {
    std::atomic<uint64_t> acquired {0};
    std::atomic<uint64_t> released {0};
    std::atomic<uint64_t> release_failures {0};
};

struct runtime_test_access;

class api_domain final {
public:
    static result<std::shared_ptr<api_domain>> load(const std::filesystem::path & path);
    template<class T>
    auto call(T && function) {
        std::lock_guard<std::mutex> lock(api_mutex_);
        return function(api_.table_);
    }
    const provider_identity & identity() const noexcept { return api_.identity(); }
    cleanup_statistics & statistics() noexcept { return cleanup_; }
    status validate(bool profile) const { return api_.validate_required_functions(profile); }
private:
    friend struct runtime_test_access;
    api_domain(std::shared_ptr<dll_module> module, qnn_api api) : module_(std::move(module)), api_(std::move(api)) {}
    std::shared_ptr<dll_module> module_;
    qnn_api api_;
    std::mutex api_mutex_;
    cleanup_statistics cleanup_;
};

struct log_tag {};
struct backend_tag {};
struct device_tag {};
struct profile_tag {};
struct context_tag {};
struct release_context { Qnn_ProfileHandle_t profile = nullptr; };

template<class T> struct handle_traits;
#define QAIRT_HANDLE_TRAIT(tag, type, operation) \
    template<> struct handle_traits<tag> { \
        using handle_type = type; \
        static Qnn_ErrorHandle_t free(const QNN_INTERFACE_VER_TYPE & api, handle_type handle, release_context) { \
            return api.operation(handle); \
        } \
    }
QAIRT_HANDLE_TRAIT(log_tag, Qnn_LogHandle_t, logFree);
QAIRT_HANDLE_TRAIT(backend_tag, Qnn_BackendHandle_t, backendFree);
QAIRT_HANDLE_TRAIT(device_tag, Qnn_DeviceHandle_t, deviceFree);
QAIRT_HANDLE_TRAIT(profile_tag, Qnn_ProfileHandle_t, profileFree);
#undef QAIRT_HANDLE_TRAIT
template<> struct handle_traits<context_tag> {
    using handle_type = Qnn_ContextHandle_t;
    static Qnn_ErrorHandle_t free(const QNN_INTERFACE_VER_TYPE & api, handle_type handle, release_context release) {
        return api.contextFree(handle, release.profile);
    }
};

template<class T>
class unique_qnn_handle final {
public:
    using handle_type = typename handle_traits<T>::handle_type;
    unique_qnn_handle() : retention_(std::make_unique<std::shared_ptr<api_domain>>()) {}
    unique_qnn_handle(const unique_qnn_handle &) = delete;
    unique_qnn_handle & operator=(const unique_qnn_handle &) = delete;
    unique_qnn_handle & operator=(unique_qnn_handle &&) = delete;
    unique_qnn_handle(unique_qnn_handle && other) noexcept :
        handle_(other.handle_), domain_(std::move(other.domain_)), release_(other.release_),
        failed_code_(other.failed_code_), retention_(std::move(other.retention_)) {
        other.handle_ = nullptr;
        other.failed_code_ = QNN_SUCCESS;
    }
    ~unique_qnn_handle() noexcept {
        if (close_raw() != QNN_SUCCESS) {
            // Owners retain the entire ancestry on failure. Pin a standalone handle's API too.
            (void) retention_.release();
        }
    }
    void adopt_into_empty(handle_type handle, std::shared_ptr<api_domain> domain, release_context release = {}) noexcept {
        GGML_ASSERT(!handle_ && handle && domain && retention_);
        handle_ = handle;
        domain_ = std::move(domain);
        *retention_ = domain_;
        release_ = release;
        failed_code_ = QNN_SUCCESS;
        ++domain_->statistics().acquired;
    }
    handle_type get() const noexcept { return handle_; }
    explicit operator bool() const noexcept { return handle_ != nullptr; }
    Qnn_ErrorHandle_t close_raw() noexcept {
        if (!handle_) { return QNN_SUCCESS; }
        if (failed_code_ != QNN_SUCCESS) { return failed_code_; }
        Qnn_ErrorHandle_t code = QNN_COMMON_ERROR_GENERAL;
        try {
            code = domain_->call([&](const auto & api) { return handle_traits<T>::free(api, handle_, release_); });
        } catch (...) {}
        if (code != QNN_SUCCESS) {
            failed_code_ = code;
            ++domain_->statistics().release_failures;
            return code;
        }
        ++domain_->statistics().released;
        handle_ = nullptr;
        retention_->reset();
        domain_.reset();
        return QNN_SUCCESS;
    }
    status close() { return check_qnn(close_raw(), error_stage::cleanup, "handleFree"); }
private:
    handle_type handle_ = nullptr;
    std::shared_ptr<api_domain> domain_;
    release_context release_;
    Qnn_ErrorHandle_t failed_code_ = QNN_SUCCESS;
    std::unique_ptr<std::shared_ptr<api_domain>> retention_;
};

class context_owner;

class runtime_owner final {
public:
    runtime_owner(const runtime_owner &) = delete;
    runtime_owner & operator=(const runtime_owner &) = delete;
    runtime_owner(runtime_owner &&) = delete;
    runtime_owner & operator=(runtime_owner &&) = delete;
    const runtime_config & config() const noexcept { return *config_; }
    const target_policy & policy() const noexcept { return *policy_; }
    const platform_snapshot & platform() const noexcept { return platform_; }
    std::shared_ptr<api_domain> domain() const noexcept { return domain_; }
    Qnn_BackendHandle_t backend() const noexcept { return backend_.get(); }
    Qnn_DeviceHandle_t device() const noexcept { return device_.get(); }
    Qnn_ProfileHandle_t profile() const noexcept { return profile_.get(); }
    const QnnGraph_Config_t ** graph_configs() noexcept { return target_configs_->graph(); }
    const QnnContext_Config_t ** context_configs() noexcept { return target_configs_->context(); }
    status close();
    bool closed() const noexcept { return closed_; }
    result<uint32_t> profile_event_count();
private:
    friend class runtime_factory;
    friend class context_owner;
    runtime_owner() = default;
    Qnn_ErrorHandle_t close_raw() noexcept;
    static void dispose(runtime_owner * owner) noexcept;
    std::shared_ptr<const runtime_config> config_;
    std::unique_ptr<target_policy> policy_;
    std::shared_ptr<api_domain> domain_;
    std::unique_ptr<target_config_storage> target_configs_;
    platform_snapshot platform_;
    uint32_t active_contexts_ = 0;
    bool closed_ = false;
    const QnnDevice_PlatformInfo_t * platform_info_ = nullptr;
    Qnn_ErrorHandle_t platform_free_error_ = QNN_SUCCESS;
    unique_qnn_handle<log_tag> log_;
    unique_qnn_handle<backend_tag> backend_;
    unique_qnn_handle<device_tag> device_;
    unique_qnn_handle<profile_tag> profile_;
};

class runtime_factory final {
public:
    static result<std::shared_ptr<runtime_owner>> create(std::shared_ptr<const runtime_config> config);
private:
    friend struct runtime_test_access;
    static result<std::shared_ptr<runtime_owner>> create(
        std::shared_ptr<const runtime_config> config, std::shared_ptr<api_domain> domain);
};

}
