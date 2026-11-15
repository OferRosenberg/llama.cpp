#pragma once

#include "ggml.h"
#include <QnnInterface.h>

#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace ggml::qairt {

enum class target_kind { htp, gpu, cpu };
enum class error_stage {
    configuration, library_load, provider_query, api_compatibility, logging,
    backend_create, platform_query, device_create, profile_create, context_create,
    graph_create, tensor_registration, op_validation, node_addition, graph_finalize,
    io_binding, graph_execute, numerical_check, binary_restore, cleanup
};

struct qairt_error {
    error_stage stage;
    target_kind target = target_kind::htp;
    std::optional<Qnn_ErrorHandle_t> qnn_error;
    std::optional<uint32_t> windows_error;
    std::string operation;
    std::string message;
};

inline qairt_error host_error(error_stage stage, const char * operation, const char * message,
                              target_kind target = target_kind::htp) {
    return { stage, target, {}, {}, operation, message };
}

inline qairt_error sdk_error(error_stage stage, const char * operation, Qnn_ErrorHandle_t code) {
    return { stage, target_kind::htp, code, {}, operation, "QNN call failed" };
}

template<class T>
class result {
public:
    static result success(T value) { return result(std::move(value)); }
    static result failure(qairt_error error) { return result(std::move(error)); }
    bool ok() const noexcept { return std::holds_alternative<T>(storage_); }
    T & value() & { GGML_ASSERT(ok()); return std::get<T>(storage_); }
    const T & value() const & { GGML_ASSERT(ok()); return std::get<T>(storage_); }
    T && value() && { GGML_ASSERT(ok()); return std::get<T>(std::move(storage_)); }
    const qairt_error & error() const & { GGML_ASSERT(!ok()); return std::get<qairt_error>(storage_); }

private:
    explicit result(T value) : storage_(std::move(value)) {}
    explicit result(qairt_error error) : storage_(std::move(error)) {}
    std::variant<T, qairt_error> storage_;
};

template<>
class result<void> {
public:
    static result success() { return result(); }
    static result failure(qairt_error error) { return result(std::move(error)); }
    bool ok() const noexcept { return !error_; }
    const qairt_error & error() const & { GGML_ASSERT(error_); return *error_; }

private:
    result() = default;
    explicit result(qairt_error error) : error_(std::move(error)) {}
    std::optional<qairt_error> error_;
};

using status = result<void>;

inline status check_qnn(Qnn_ErrorHandle_t code, error_stage stage, const char * operation) {
    return code == QNN_SUCCESS ? status::success() : status::failure(sdk_error(stage, operation, code));
}

}
