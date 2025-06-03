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

/*
torch::ScalarType dtype(std::string_view type_str) {
    if        (type_str == "u8") {
        return torch::kUInt8 ;
    } else if (type_str == "i8") {
        return torch::kInt8 ;
    } else if (type_str == "u16") {
        return torch::kUInt16 ;
    } else if (type_str == "i16") {
        return torch::kInt16 ;
    } else if (type_str == "u32") {
        return torch::kUInt32 ;
    } else if (type_str == "i32") {
        return torch::kInt32 ;
    } else if (type_str == "u64") {
        return torch::kUInt64 ;
    } else if (type_str == "i64") {
        return torch::kInt64 ;
    } else if (type_str == "f16") {
        return torch::kFloat16 ;
    } else if (type_str == "f32") {
        return torch::kFloat32 ;
    } else if (type_str == "f64") {
        return torch::kFloat64 ;
    } else {
        throw std::runtime_error(std::format("Invalid type: {}", type_str));
    }
}
*/

/*
extern "C" int GetTypeId(char const* type_str) {
    return static_cast<int>(dtype(type_str));
}
*/

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

struct Tensor* NewTensorFromBlob(void* blob, size_t size, size_t* shape, size_t dims, torch::ScalarType dtype) {
    try {
	std::vector<at::IntArrayRef::value_type> tensor_shape = {static_cast<at::IntArrayRef::value_type>(size)};
	tensor_shape.insert(tensor_shape.end(), shape, shape + dims);
        auto options = torch::TensorOptions().dtype(dtype);

        return new Tensor(blob, tensor_shape, options);
    } catch (const std::exception& e) {
	std::cerr << "Exception - " << e.what() << std::endl ;
    }
    return nullptr ;
}

#define NEW_TENSOR_FN(dtype) \
extern "C" struct Tensor* NewTensor##dtype(void* blob, size_t size, size_t* shape, size_t dims) { \
    return NewTensorFromBlob(blob, size, shape, dims, torch::k##dtype); \
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


extern "C" void FreeTensor(struct Tensor* tensor) {
    delete tensor ;
}

extern "C" size_t Flatten(struct Tensor* tensor, void* result_buffer, size_t buffer_size, size_t* shape, size_t maxDim) {
    try {
        tensor->to(torch::kCPU);

        if (shape && maxDim > 0) {
            if (tensor->tensor().dim() >= maxDim) {
                throw std::runtime_error(std::format("Insufficient dimensions {} given, want {}", maxDim, tensor->tensor().dim()));
            }

            std::fill(shape, shape + maxDim, size_t{0});

            for(auto i = 0; i < tensor->tensor().dim(); ++i) {
                shape[i] = tensor->tensor().size(i);
            }
        }

        auto output_flat = tensor->tensor().view({-1});

        size_t result_size = output_flat.size(0);

        if (result_size <= buffer_size) {
            auto output_cpu = output_flat.to(torch::kCPU);
            std::memcpy(result_buffer, output_cpu.data_ptr<float>(), result_size * sizeof(float));
        } else {
            throw std::runtime_error(std::format("Insufficient buffer size {} given, want {}", buffer_size, result_size));
            return 0;
        }
    } catch (const std::exception& e) {
	std::cerr << "Exception - " << e.what() << std::endl ;
	return 0;
    }

    return tensor->tensor().dim();
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
