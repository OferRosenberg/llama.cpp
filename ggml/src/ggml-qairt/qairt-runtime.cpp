#include "qairt-runtime.h"
#include "ggml-impl.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace ggml::qairt {

struct dll_module::native_module {
    HMODULE handle = nullptr;
};

dll_module::dll_module() : native_(std::make_unique<native_module>()) {}

dll_module::~dll_module() noexcept {
    if (native_->handle) { FreeLibrary(native_->handle); }
}

result<std::shared_ptr<dll_module>> dll_module::load(const std::filesystem::path & path) {
    if (!path.is_absolute()) {
        return result<std::shared_ptr<dll_module>>::failure(host_error(error_stage::library_load, "LoadLibraryExW", "Provider path must be absolute"));
    }
    auto module = std::shared_ptr<dll_module>(new dll_module());
    module->path_ = std::filesystem::weakly_canonical(path);
    module->native_->handle = LoadLibraryExW(module->path_.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!module->native_->handle) {
        const auto windows_error = GetLastError();
        auto error = host_error(error_stage::library_load, "LoadLibraryExW", "Could not load provider or its dependencies");
        error.windows_error = windows_error;
        return result<std::shared_ptr<dll_module>>::failure(std::move(error));
    }
    return result<std::shared_ptr<dll_module>>::success(std::move(module));
}

result<dll_module::symbol_address> dll_module::resolve_address(const char * symbol) const {
    auto address = GetProcAddress(native_->handle, symbol);
    if (!address) {
        const auto windows_error = GetLastError();
        auto error = host_error(error_stage::provider_query, symbol, "Provider export missing");
        error.windows_error = windows_error;
        return result<symbol_address>::failure(std::move(error));
    }
    return result<symbol_address>::success(reinterpret_cast<symbol_address>(address));
}

result<qnn_api> qnn_api::select_provider(const dll_module & module) {
    using get_providers = Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *);
    auto symbol = module.resolve<get_providers>("QnnInterface_getProviders");
    if (!symbol.ok()) { return result<qnn_api>::failure(symbol.error()); }
    const QnnInterface_t ** providers = nullptr;
    uint32_t count = 0;
    const auto code = symbol.value()(&providers, &count);
    if (code != QNN_SUCCESS) { return result<qnn_api>::failure(sdk_error(error_stage::provider_query, "QnnInterface_getProviders", code)); }
    auto selected = from_providers(providers, count);
    if (selected.ok()) { selected.value().identity_.module_path = module.canonical_path(); }
    return selected;
}

result<qnn_api> qnn_api::from_providers(const QnnInterface_t * const * providers, uint32_t count) {
    if (!providers || !count || count > 128) {
        return result<qnn_api>::failure(host_error(error_stage::provider_query, "select_provider", "Empty or invalid provider list"));
    }
    const QnnInterface_t * selected = nullptr;
    for (uint32_t i = 0; i < count; ++i) {
        const auto * provider = providers[i];
        if (!provider || provider->apiVersion.coreApiVersion.major != QNN_API_VERSION_MAJOR ||
            provider->apiVersion.coreApiVersion.minor < QNN_API_VERSION_MINOR) { continue; }
        if (!selected || provider->apiVersion.coreApiVersion.minor < selected->apiVersion.coreApiVersion.minor ||
            (provider->apiVersion.coreApiVersion.minor == selected->apiVersion.coreApiVersion.minor &&
             provider->backendId < selected->backendId)) {
            selected = provider;
        }
    }
    if (!selected) {
        return result<qnn_api>::failure(host_error(error_stage::api_compatibility, "select_provider", "No provider satisfies the header core API version"));
    }
    qnn_api api;
    api.table_ = selected->QNN_INTERFACE_VER_NAME;
    api.identity_.backend_id = selected->backendId;
    api.identity_.version = selected->apiVersion;
    api.identity_.provider_name = selected->providerName ? selected->providerName : "unnamed";
    return result<qnn_api>::success(std::move(api));
}

status qnn_api::validate_required_functions(bool profile) const {
    if (!table_.logCreate || !table_.logFree || !table_.backendCreate || !table_.backendFree ||
        !table_.deviceCreate || !table_.deviceFree || !table_.contextCreate || !table_.contextFree ||
        !table_.graphCreate || !table_.graphAddNode || !table_.graphFinalize || !table_.graphExecute ||
        !table_.tensorCreateGraphTensor || !table_.backendValidateOpConfig) {
        return status::failure(host_error(error_stage::api_compatibility, "interface_table", "Required M0 function missing"));
    }
    if (profile && (!table_.profileCreate || !table_.profileFree || !table_.profileGetEvents)) {
        return status::failure(host_error(error_stage::api_compatibility, "profile", "Profiling functions missing"));
    }
    return status::success();
}

result<std::shared_ptr<api_domain>> api_domain::load(const std::filesystem::path & path) {
    auto module = dll_module::load(path);
    if (!module.ok()) { return result<std::shared_ptr<api_domain>>::failure(module.error()); }
    auto api = qnn_api::select_provider(*module.value());
    if (!api.ok()) { return result<std::shared_ptr<api_domain>>::failure(api.error()); }
    return result<std::shared_ptr<api_domain>>::success(std::shared_ptr<api_domain>(
        new api_domain(std::move(module).value(), std::move(api).value())));
}

static void log_callback(const char * format, QnnLog_Level_t level, uint64_t, va_list args) noexcept {
    if (!format) { return; }
    char message[2048] {};
    va_list copy;
    va_copy(copy, args);
    const int formatted = std::vsnprintf(message, sizeof(message), format, copy);
    va_end(copy);
    message[sizeof(message) - 1] = '\0';
    if (formatted < 0) { return; }
    auto severity = GGML_LOG_LEVEL_INFO;
    if (level == QNN_LOG_LEVEL_ERROR) { severity = GGML_LOG_LEVEL_ERROR; }
    else if (level == QNN_LOG_LEVEL_WARN) { severity = GGML_LOG_LEVEL_WARN; }
    else if (level == QNN_LOG_LEVEL_DEBUG || level == QNN_LOG_LEVEL_VERBOSE) { severity = GGML_LOG_LEVEL_DEBUG; }
    try { ggml_log_internal(severity, "qairt: %s\n", message); } catch (...) {}
}

void runtime_owner::dispose(runtime_owner * owner) noexcept {
    if (owner->close_raw() == QNN_SUCCESS) {
        delete owner;
    } else {
        // Keep this already allocated owner and all its dependencies alive without allocating.
        std::fprintf(stderr, "qairt: cleanup failed; retaining runtime ancestry until process exit\n");
    }
}

Qnn_ErrorHandle_t runtime_owner::close_raw() noexcept {
    if (closed_) { return QNN_SUCCESS; }
    if (active_contexts_) { return QNN_COMMON_ERROR_RESOURCE_UNAVAILABLE; }
    if (platform_info_) {
        if (platform_free_error_ != QNN_SUCCESS) { return platform_free_error_; }
        try {
            platform_free_error_ = domain_->call([&](const auto & api) { return api.deviceFreePlatformInfo(log_.get(), platform_info_); });
        } catch (...) { platform_free_error_ = QNN_COMMON_ERROR_GENERAL; }
        if (platform_free_error_ != QNN_SUCCESS) {
            ++domain_->statistics().release_failures;
            return platform_free_error_;
        }
        platform_info_ = nullptr;
    }
    auto code = profile_.close_raw();
    if (code != QNN_SUCCESS) { return code; }
    code = device_.close_raw();
    if (code != QNN_SUCCESS) { return code; }
    code = backend_.close_raw();
    if (code != QNN_SUCCESS) { return code; }
    code = log_.close_raw();
    if (code != QNN_SUCCESS) { return code; }
    closed_ = true;
    return QNN_SUCCESS;
}

status runtime_owner::close() {
    if (active_contexts_) { return status::failure(host_error(error_stage::cleanup, "runtime.close", "Active contexts remain")); }
    return check_qnn(close_raw(), error_stage::cleanup, "runtime.close");
}

result<uint32_t> runtime_owner::profile_event_count() {
    if (closed_ || !profile_) { return result<uint32_t>::failure(host_error(error_stage::profile_create, "profileGetEvents", "No live profile")); }
    const QnnProfile_EventId_t * events = nullptr;
    uint32_t count = 0;
    auto code = domain_->call([&](const auto & api) { return api.profileGetEvents(profile_.get(), &events, &count); });
    if (code != QNN_SUCCESS) { return result<uint32_t>::failure(sdk_error(error_stage::profile_create, "profileGetEvents", code)); }
    if (count && !events) { return result<uint32_t>::failure(host_error(error_stage::profile_create, "profileGetEvents", "Invalid event array")); }
    return result<uint32_t>::success(count);
}

result<std::shared_ptr<runtime_owner>> runtime_factory::create(std::shared_ptr<const runtime_config> config) {
    if (!config) { return result<std::shared_ptr<runtime_owner>>::failure(host_error(error_stage::configuration, "runtime.create", "Null configuration")); }
    auto checked = config_parser::validate(*config);
    if (!checked.ok()) { return result<std::shared_ptr<runtime_owner>>::failure(checked.error()); }
    auto policy = target_catalog::create_policy(config->target);
    if (!policy.ok()) { return result<std::shared_ptr<runtime_owner>>::failure(policy.error()); }
    auto domain = api_domain::load(config->library_directory / policy.value()->descriptor().provider_filename);
    if (!domain.ok()) { return result<std::shared_ptr<runtime_owner>>::failure(domain.error()); }
    return create(std::move(config), std::move(domain).value());
}

result<std::shared_ptr<runtime_owner>> runtime_factory::create(
        std::shared_ptr<const runtime_config> config, std::shared_ptr<api_domain> domain) {
    using runtime_result = result<std::shared_ptr<runtime_owner>>;
    if (!config || !domain) { return runtime_result::failure(host_error(error_stage::configuration, "runtime.create", "Null configuration or API domain")); }
    auto checked = config_parser::validate(*config);
    if (!checked.ok()) { return runtime_result::failure(checked.error()); }
    auto policy = target_catalog::create_policy(config->target);
    if (!policy.ok()) { return runtime_result::failure(policy.error()); }
    checked = domain->validate(config->enable_profile);
    if (!checked.ok()) { return runtime_result::failure(checked.error()); }
    auto owner = std::shared_ptr<runtime_owner>(new runtime_owner(), runtime_owner::dispose);
    owner->config_ = std::move(config);
    owner->domain_ = domain;
    owner->policy_ = std::move(policy).value();
    auto configs = owner->policy_->build_configs();
    if (!configs.ok()) { return runtime_result::failure(configs.error()); }
    owner->target_configs_ = std::move(configs).value();

    Qnn_LogHandle_t log = nullptr;
    const auto log_level = owner->config_->verbose_diagnostics ? QNN_LOG_LEVEL_DEBUG : QNN_LOG_LEVEL_WARN;
    auto code = domain->call([&](const auto & api) { return api.logCreate(log_callback, log_level, &log); });
    if (code != QNN_SUCCESS) { return runtime_result::failure(sdk_error(error_stage::logging, "logCreate", code)); }
    owner->log_.adopt_into_empty(log, domain);
    Qnn_BackendHandle_t backend = nullptr;
    code = domain->call([&](const auto & api) { return api.backendCreate(log, owner->target_configs_->backend(), &backend); });
    if (code != QNN_SUCCESS) { return runtime_result::failure(sdk_error(error_stage::backend_create, "backendCreate", code)); }
    owner->backend_.adopt_into_empty(backend, domain);

    bool query_platform = domain->call([](const auto & api) { return api.deviceGetPlatformInfo && api.deviceFreePlatformInfo; });
    if (query_platform) {
        code = domain->call([&](const auto & api) { return api.deviceGetPlatformInfo(log, &owner->platform_info_); });
        if (code == QNN_SUCCESS && owner->platform_info_) {
            auto snapshot = owner->policy_->read_platform(*owner->platform_info_);
            owner->platform_free_error_ = domain->call([&](const auto & api) { return api.deviceFreePlatformInfo(log, owner->platform_info_); });
            if (owner->platform_free_error_ != QNN_SUCCESS) {
                ++domain->statistics().release_failures;
                return runtime_result::failure(sdk_error(error_stage::cleanup, "deviceFreePlatformInfo", owner->platform_free_error_));
            }
            owner->platform_info_ = nullptr;
            if (!snapshot.ok()) { return runtime_result::failure(snapshot.error()); }
            owner->platform_ = snapshot.value();
        } else if (code != QNN_SUCCESS) {
            owner->platform_info_ = nullptr;
            if (code != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) { return runtime_result::failure(sdk_error(error_stage::platform_query, "deviceGetPlatformInfo", code)); }
        } else {
            return runtime_result::failure(host_error(error_stage::platform_query, "deviceGetPlatformInfo", "Null platform metadata"));
        }
    }
    Qnn_DeviceHandle_t device = nullptr;
    code = domain->call([&](const auto & api) { return api.deviceCreate(log, owner->target_configs_->device(), &device); });
    if (code != QNN_SUCCESS) { return runtime_result::failure(sdk_error(error_stage::device_create, "deviceCreate", code)); }
    owner->device_.adopt_into_empty(device, domain);
    if (owner->config_->enable_profile) {
        Qnn_ProfileHandle_t profile = nullptr;
        code = domain->call([&](const auto & api) { return api.profileCreate(backend, QNN_PROFILE_LEVEL_BASIC, &profile); });
        if (code != QNN_SUCCESS) { return runtime_result::failure(sdk_error(error_stage::profile_create, "profileCreate", code)); }
        owner->profile_.adopt_into_empty(profile, domain);
    }
    return runtime_result::success(std::move(owner));
}

}
