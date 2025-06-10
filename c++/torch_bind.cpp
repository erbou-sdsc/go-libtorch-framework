#include <algorithm>
#include <array>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <torch/torch.h>
#include "torch_bind.h"
#include "torch.h"

torch::ScalarType str2dtype(std::string_view type_str) {
   if (type_str.size() >= 2) {
        switch(type_str[0]) {
            case 'f':
                switch(type_str[1]) {
                    case '1':
                        return torch::kFloat16;
                    case '3':
                        return torch::kFloat32;
                    case '6':
                        return torch::kFloat64;
                    default: break ;
                }
                break;
            case 'i':
                switch(type_str[1]) {
                    case '1':
                        return torch::kInt16;
                    case '3':
                        return torch::kInt32;
                    case '6':
                        return torch::kInt64;
                    default: break ;
                }
                break;
            case 'u':
                switch(type_str[1]) {
                    case '1':
                        return torch::kUInt16;
                    case '3':
                        return torch::kUInt32;
                    case '6':
                        return torch::kUInt64;
                    default: break ;
                }
                break;
        }
    }
    throw std::runtime_error(std::format("Invalid type: {}", type_str));
}

char const* dtype2str(torch::ScalarType dtype) {
    switch(dtype) {
        case torch::kFloat16:
            return "f16";
        case torch::kFloat32:
            return "f32";
        case torch::kFloat64:
            return "f64";
        case torch::kInt8:
            return "i8";
        case torch::kInt16:
            return "i16";
        case torch::kInt32:
            return "i32";
        case torch::kInt64:
            return "i64";
        case torch::kUInt8:
            return "u8";
        case torch::kUInt16:
            return "u16";
        case torch::kUInt32:
            return "u32";
        case torch::kUInt64:
            return "u64";
        default:
            return "bad";
            //throw std::runtime_error(std::format("Invalid type: {}", static_cast<int>(dtype)));
    }
}

extern "C" int GetTypeId(char const* type_str) {
    return static_cast<int>(str2dtype(type_str));
}

extern "C" Model* NewModel(char const* model, char const* options) {
    try {
        return model_factory::newModel(model, std::string_view{options});
    } catch (const std::exception& e) {
        std::cerr << "Exception - " << e.what() << std::endl ;
        return nullptr;
    }
}

extern "C" void FreeModel(struct Model* model) {
    delete model;
}

struct Tensor* NewTensorFromBlob(void* blob, size_t* shape, size_t dims, torch::ScalarType dtype) {
    try {
	    std::vector<at::IntArrayRef::value_type> tensor_shape(shape, shape + dims);
        auto options = torch::TensorOptions().dtype(dtype);
        return new Tensor(blob, tensor_shape, options);
    } catch (const std::exception& e) {
	    std::cerr << "Exception - " << e.what() << std::endl ;
    }
    return nullptr ;
}

#define NEW_TENSOR_FN(dtype) \
extern "C" struct Tensor* NewTensor##dtype(void* blob, size_t* shape, size_t dims) { \
    return NewTensorFromBlob(blob, shape, dims, torch::k##dtype); \
}

NEW_TENSOR_FN(Int8)
NEW_TENSOR_FN(Int16)
NEW_TENSOR_FN(Int32)
NEW_TENSOR_FN(Int64)
NEW_TENSOR_FN(UInt8)
NEW_TENSOR_FN(UInt16)
NEW_TENSOR_FN(UInt32)
NEW_TENSOR_FN(UInt64)
NEW_TENSOR_FN(Float32)
NEW_TENSOR_FN(Float64)
#undef NEW_TENSOR_FN

extern "C" size_t Dim(struct Tensor const* tensor) {
    return tensor ? static_cast<size_t>(tensor->tensor().dim()): 0 ;
}

extern "C" int64_t const* Shape(struct Tensor const* tensor) {
    return tensor ? tensor->tensor().sizes().data(): nullptr ;
}

extern "C" void FreeTensor(struct Tensor* tensor) {
    delete tensor ;
}

extern "C" size_t ElemSize(struct Tensor const* tensor) {
    return tensor->tensor().element_size();
}

extern "C" size_t NumElem(struct Tensor const* tensor) {
    return tensor->tensor().numel();
}

extern "C" size_t NumBytes(struct Tensor const* tensor) {
    return tensor->tensor().numel() * tensor->tensor().element_size();
}

extern "C" char const* DType(struct Tensor const* tensor) {
    return dtype2str(tensor->tensor().scalar_type());
    //return tensor->tensor().dtype().name().data();
}

extern "C" size_t Bytes(struct Tensor const* tensor, void* result_buffer, size_t buffer_size) {
    try {
        auto result_size = NumBytes(tensor);
        if (result_size < buffer_size) {
            throw std::runtime_error(std::format("Insufficient buffer size {} given, want {}", buffer_size, result_size));
        }

        // Make a copy of the tensor to the CPU
        auto tensor_cpu = tensor->tensor().to(torch::kCPU, false, false).contiguous();
        std::memcpy(result_buffer, tensor_cpu.data_ptr(), result_size);
        return result_size;
    } catch (const std::exception& e) {
	    std::cerr << "Exception - " << e.what() << std::endl ;
	    return 0;
    }
}

extern "C" struct Tensor* Infer(struct Model* model, struct Tensor* data) {
    if (model == nullptr || data == nullptr) {
        return nullptr ;
    }

    try {
        c10::InferenceMode guard;

        data->to(device_manager::device());

        model->module().eval();

        return new Tensor{model->forward(*data)};

    } catch (const std::exception& e) {
	std::cerr << "Exception - " << e.what() << std::endl ;
	return nullptr;
    }
}

extern "C" void Train(struct Model* model, struct Tensor* data, struct Tensor const** targets, int num_targets, int epochs) {
    if (model == nullptr || data == nullptr || targets == nullptr || epochs == 0) {
        return ;
    }

    try {
        data->to(device_manager::device());
        std::vector<torch::Tensor const*> vtargets ;
        for (int i = 0; i < num_targets; ++i) {
            targets[i]->tensor().to(device_manager::device());
            vtargets.push_back(&targets[i]->tensor());
        }
        model->module().train();

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            model->optimizer()->zero_grad();
            auto output = model->forward(*data);

            auto loss = model->prepare_targets(output, vtargets);

            loss.backward();
            model->optimizer()->step();
    
            //std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "] Loss: " << loss.item<float>() << std::endl;
        }
    } catch (const std::exception& e) {
	    std::cerr << "Exception - " << e.what() << std::endl ;
    }
}
