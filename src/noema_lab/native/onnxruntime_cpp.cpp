#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ort/onnxruntime_c_api.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

class OrtError : public std::runtime_error {
public:
    explicit OrtError(const std::string& message) : std::runtime_error(message) {}
};

#ifdef _WIN32
std::wstring utf8_to_windows_path(const std::string& value) {
    if (value.empty()) {
        return std::wstring();
    }
    const int required = MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        value.data(),
        static_cast<int>(value.size()),
        nullptr,
        0
    );
    if (required <= 0) {
        throw OrtError(
            "Could not decode filesystem path as UTF-8 (Windows error "
            + std::to_string(GetLastError()) + ")"
        );
    }
    std::wstring result(static_cast<std::size_t>(required), L'\0');
    const int written = MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        value.data(),
        static_cast<int>(value.size()),
        result.data(),
        required
    );
    if (written != required) {
        throw OrtError(
            "Could not convert filesystem path to UTF-16 (Windows error "
            + std::to_string(GetLastError()) + ")"
        );
    }
    return result;
}
#endif

class SharedLibrary {
public:
    explicit SharedLibrary(const std::string& path) {
#ifdef _WIN32
        const std::wstring native_path = utf8_to_windows_path(path);
        handle_ = LoadLibraryW(native_path.c_str());
        if (!handle_) {
            throw OrtError(
                "Could not load ONNX Runtime library " + path
                + " (Windows error " + std::to_string(GetLastError()) + ")"
            );
        }
#else
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle_) {
            const char* error = dlerror();
            throw OrtError("Could not load ONNX Runtime library " + path + ": " + (error ? error : "unknown error"));
        }
#endif
    }

    ~SharedLibrary() {
        if (handle_) {
#ifdef _WIN32
            FreeLibrary(handle_);
#else
            dlclose(handle_);
#endif
        }
    }

    void* symbol(const char* name) const {
#ifdef _WIN32
        FARPROC pointer = GetProcAddress(handle_, name);
        if (!pointer) {
            throw OrtError(
                std::string("Could not resolve ONNX Runtime symbol ") + name
                + " (Windows error " + std::to_string(GetLastError()) + ")"
            );
        }
        return reinterpret_cast<void*>(pointer);
#else
        dlerror();
        void* pointer = dlsym(handle_, name);
        const char* error = dlerror();
        if (error || !pointer) {
            throw OrtError(std::string("Could not resolve ONNX Runtime symbol ") + name + ": " + (error ? error : "unknown error"));
        }
        return pointer;
#endif
    }

private:
#ifdef _WIN32
    HMODULE handle_ = nullptr;
#else
    void* handle_ = nullptr;
#endif
};

using OrtGetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)();

const OrtApi* load_api(const SharedLibrary& library) {
    auto get_api_base = reinterpret_cast<OrtGetApiBaseFn>(library.symbol("OrtGetApiBase"));
    const OrtApiBase* base = get_api_base();
    if (!base) {
        throw OrtError("OrtGetApiBase returned null");
    }
    const OrtApi* api = base->GetApi(ORT_API_VERSION);
    if (!api) {
        throw OrtError("ONNX Runtime C API version is not available");
    }
    return api;
}

void check_status(const OrtApi* api, OrtStatus* status, const std::string& context) {
    if (!status) {
        return;
    }
    const char* message = api->GetErrorMessage(status);
    std::string text = context + ": " + (message ? message : "unknown ONNX Runtime error");
    api->ReleaseStatus(status);
    throw OrtError(text);
}

template <typename T>
std::size_t element_count(const std::vector<T>& dims) {
    std::size_t count = 1;
    for (const T dim : dims) {
        if (dim < 0) {
            throw OrtError("Output tensor has an unresolved negative dimension");
        }
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

std::string allocated_name(const OrtApi* api, OrtSession* session, OrtAllocator* allocator, std::size_t index, bool output) {
    char* raw = nullptr;
    OrtStatus* status = output
        ? api->SessionGetOutputName(session, index, allocator, &raw)
        : api->SessionGetInputName(session, index, allocator, &raw);
    check_status(api, status, output ? "SessionGetOutputName" : "SessionGetInputName");
    std::string name = raw ? std::string(raw) : std::string();
    if (raw) {
        check_status(api, api->AllocatorFree(allocator, raw), "AllocatorFree");
    }
    return name;
}

ONNXTensorElementDataType numpy_dtype_to_ort(const py::dtype& dtype) {
    if (py::isinstance<py::dtype>(dtype)) {
        if (dtype.is(py::dtype::of<float>())) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
        if (dtype.is(py::dtype::of<int64_t>())) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        }
    }
    throw OrtError("ONNX Runtime C++ bridge supports only float32 and int64 tensor inputs");
}

py::array ort_value_to_numpy(const OrtApi* api, OrtValue* value) {
    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    check_status(api, api->GetTensorTypeAndShape(value, &shape_info), "GetTensorTypeAndShape");

    ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    try {
        check_status(api, api->GetTensorElementType(shape_info, &element_type), "GetTensorElementType");
        std::size_t rank = 0;
        check_status(api, api->GetDimensionsCount(shape_info, &rank), "GetDimensionsCount");
        std::vector<int64_t> dims(rank);
        if (rank) {
            check_status(api, api->GetDimensions(shape_info, dims.data(), dims.size()), "GetDimensions");
        }
        void* data = nullptr;
        check_status(api, api->GetTensorMutableData(value, &data), "GetTensorMutableData");
        const std::size_t count = element_count(dims);
        std::vector<py::ssize_t> shape;
        shape.reserve(dims.size());
        for (const int64_t dim : dims) {
            shape.push_back(static_cast<py::ssize_t>(dim));
        }

        if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            py::array_t<float> out(shape);
            std::memcpy(out.mutable_data(), data, count * sizeof(float));
            api->ReleaseTensorTypeAndShapeInfo(shape_info);
            return out;
        }
        if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            py::array_t<int64_t> out(shape);
            std::memcpy(out.mutable_data(), data, count * sizeof(int64_t));
            api->ReleaseTensorTypeAndShapeInfo(shape_info);
            return out;
        }
    } catch (...) {
        api->ReleaseTensorTypeAndShapeInfo(shape_info);
        throw;
    }

    api->ReleaseTensorTypeAndShapeInfo(shape_info);
    throw OrtError("ONNX Runtime C++ bridge supports only float32 and int64 tensor outputs");
}

}  // namespace

class CppOrtSession {
public:
    CppOrtSession(std::string library_path, std::string model_path, std::string provider, int intra_op_num_threads)
        : library_(std::move(library_path)), api_(load_api(library_)), provider_(std::move(provider)), model_path_(std::move(model_path)) {
        if (provider_.empty()) {
            provider_ = "CPUExecutionProvider";
        }
        if (provider_ != "CPUExecutionProvider" && provider_ != "CPU") {
            throw OrtError("ONNX Runtime C++ bridge currently supports CPUExecutionProvider only; requested " + provider_);
        }
        check_status(api_, api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "noema_onnxruntime_cpp", &env_), "CreateEnv");
        check_status(api_, api_->CreateSessionOptions(&session_options_), "CreateSessionOptions");
        if (intra_op_num_threads > 0) {
            check_status(api_, api_->SetIntraOpNumThreads(session_options_, intra_op_num_threads), "SetIntraOpNumThreads");
        }
        check_status(api_, api_->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL), "SetSessionGraphOptimizationLevel");
#ifdef _WIN32
        const std::wstring native_model_path = utf8_to_windows_path(model_path_);
        check_status(api_, api_->CreateSession(env_, native_model_path.c_str(), session_options_, &session_), "CreateSession");
#else
        check_status(api_, api_->CreateSession(env_, model_path_.c_str(), session_options_, &session_), "CreateSession");
#endif
        check_status(api_, api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_), "CreateCpuMemoryInfo");
        OrtAllocator* allocator = nullptr;
        check_status(api_, api_->GetAllocatorWithDefaultOptions(&allocator), "GetAllocatorWithDefaultOptions");

        std::size_t input_count = 0;
        std::size_t output_count = 0;
        check_status(api_, api_->SessionGetInputCount(session_, &input_count), "SessionGetInputCount");
        check_status(api_, api_->SessionGetOutputCount(session_, &output_count), "SessionGetOutputCount");
        for (std::size_t index = 0; index < input_count; ++index) {
            input_names_.push_back(allocated_name(api_, session_, allocator, index, false));
        }
        for (std::size_t index = 0; index < output_count; ++index) {
            output_names_.push_back(allocated_name(api_, session_, allocator, index, true));
        }
    }

    ~CppOrtSession() {
        if (memory_info_) api_->ReleaseMemoryInfo(memory_info_);
        if (session_) api_->ReleaseSession(session_);
        if (session_options_) api_->ReleaseSessionOptions(session_options_);
        if (env_) api_->ReleaseEnv(env_);
    }

    py::list run(py::dict inputs) {
        std::vector<py::array> keepalive;
        std::vector<OrtValue*> input_values;
        std::vector<const char*> input_names;
        keepalive.reserve(input_names_.size());
        input_values.reserve(input_names_.size());
        input_names.reserve(input_names_.size());

        for (const auto& name : input_names_) {
            py::object object = inputs[py::str(name)];
            py::array array = py::array::ensure(object, py::array::c_style | py::array::forcecast);
            if (!array) {
                throw OrtError("Input " + name + " is not a NumPy array");
            }
            ONNXTensorElementDataType type = numpy_dtype_to_ort(array.dtype());
            py::buffer_info info = array.request();
            std::vector<int64_t> dims;
            dims.reserve(static_cast<std::size_t>(info.ndim));
            for (const auto dim : info.shape) {
                dims.push_back(static_cast<int64_t>(dim));
            }
            OrtValue* value = nullptr;
            check_status(
                api_,
                api_->CreateTensorWithDataAsOrtValue(
                    memory_info_,
                    info.ptr,
                    static_cast<std::size_t>(info.size) * static_cast<std::size_t>(info.itemsize),
                    dims.data(),
                    dims.size(),
                    type,
                    &value
                ),
                "CreateTensorWithDataAsOrtValue"
            );
            keepalive.push_back(array);
            input_values.push_back(value);
            input_names.push_back(name.c_str());
        }

        std::vector<const char*> output_names;
        std::vector<OrtValue*> output_values(output_names_.size(), nullptr);
        output_names.reserve(output_names_.size());
        for (const auto& name : output_names_) {
            output_names.push_back(name.c_str());
        }

        try {
            check_status(
                api_,
                api_->Run(
                    session_,
                    nullptr,
                    input_names.data(),
                    input_values.data(),
                    input_values.size(),
                    output_names.data(),
                    output_names.size(),
                    output_values.data()
                ),
                "Run"
            );
        } catch (...) {
            for (auto* value : input_values) {
                api_->ReleaseValue(value);
            }
            for (auto* value : output_values) {
                if (value) api_->ReleaseValue(value);
            }
            throw;
        }

        py::list outputs;
        try {
            for (auto* value : output_values) {
                outputs.append(ort_value_to_numpy(api_, value));
            }
        } catch (...) {
            for (auto* value : input_values) {
                api_->ReleaseValue(value);
            }
            for (auto* value : output_values) {
                if (value) api_->ReleaseValue(value);
            }
            throw;
        }
        for (auto* value : input_values) {
            api_->ReleaseValue(value);
        }
        for (auto* value : output_values) {
            if (value) api_->ReleaseValue(value);
        }
        return outputs;
    }

    std::vector<std::string> input_names() const { return input_names_; }
    std::vector<std::string> output_names() const { return output_names_; }
    std::string provider() const { return provider_ == "CPU" ? "CPUExecutionProvider" : provider_; }
    std::string model_path() const { return model_path_; }

private:
    SharedLibrary library_;
    const OrtApi* api_ = nullptr;
    std::string provider_;
    std::string model_path_;
    OrtEnv* env_ = nullptr;
    OrtSessionOptions* session_options_ = nullptr;
    OrtSession* session_ = nullptr;
    OrtMemoryInfo* memory_info_ = nullptr;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

PYBIND11_MODULE(_onnxruntime_cpp, m) {
    py::register_exception<OrtError>(m, "OrtError");
    py::class_<CppOrtSession>(m, "Session")
        .def(py::init<std::string, std::string, std::string, int>(), py::arg("library_path"), py::arg("model_path"), py::arg("provider") = "CPUExecutionProvider", py::arg("intra_op_num_threads") = 0)
        .def("run", &CppOrtSession::run)
        .def_property_readonly("input_names", &CppOrtSession::input_names)
        .def_property_readonly("output_names", &CppOrtSession::output_names)
        .def_property_readonly("provider", &CppOrtSession::provider)
        .def_property_readonly("model_path", &CppOrtSession::model_path);
}
