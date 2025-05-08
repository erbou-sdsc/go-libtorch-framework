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

class tensor {
public:
    tensor(void* blob, at::IntArrayRef sizes, torch::TensorOptions& options) : tensor_{torch::from_blob(blob, sizes, options).to(device_manager::device())} {}
    tensor(tensor const&) = delete;
    tensor() = delete;

    operator torch::Tensor& () {
        return tensor_ ;
    }

    operator torch::Tensor const& () const {
        return tensor_ ;
    }

private:
    torch::Tensor tensor_ ;
};

extern "C" int GetTypeId(char const* type_str) {
    return static_cast<int>(dtype(type_str));
}

extern "C" basic_model* NewModel(char const* model, char const* options) {
    try {
        return model_factory::newModel(model, std::string_view{options});
    } catch (const std::exception& e) {
        std::cerr << "Exception - " << e.what() << std::endl ;
        return nullptr;
    }
}

extern "C" void FreeModel(struct basic_model* model) {
    delete model;
}

extern "C" struct tensor* NewTensorFromBlob(uint8_t* blob, int64_t size, int64_t* shape, int64_t dims, int dtype) {
    try {
	std::vector<at::IntArrayRef::value_type> tensor_shape = {static_cast<at::IntArrayRef::value_type>(size)};
	tensor_shape.insert(tensor_shape.end(), shape, shape + dims);
        auto options = torch::TensorOptions().dtype(static_cast<torch::ScalarType>(dtype));

        return new tensor(blob, tensor_shape, options);
    } catch (const std::exception& e) {
	std::cerr << "Exception - " << e.what() << std::endl ;
    }
    return nullptr ;
}

extern "C" void FreeTensor(struct tensor* tensor) {
    delete tensor ;
}

extern "C" size_t Infer(struct basic_model* model, struct tensor const* data, void* result_buffer, size_t buffer_size) {
    try {
        c10::InferenceMode guard;

        model->eval();

        auto output = model->forward(*data);

        auto output_flat = output.view({-1});

        size_t result_size = output_flat.size(0);

        if (result_size <= buffer_size) {
            auto output_cpu = output_flat.to(torch::kCPU);
            std::memcpy(result_buffer, output_cpu.data_ptr<float>(), result_size * sizeof(float));
        } else {
	    std::cerr << "Insufficient buffer size " << buffer_size << ", " << result_size << " is needed" << std::endl ;
            return 0;
        }

        return result_size;
    } catch (const std::exception& e) {
	std::cerr << "Exception - " << e.what() << std::endl ;
	return 0;
    }
}


extern "C" void Train(struct basic_model* model, struct tensor const* data, struct tensor const* target, int epochs) {
    try {
        model->train();

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            model->optimizer()->zero_grad();
            auto output = model->forward(*data);
            auto loss = torch::nll_loss(output, *target);
    
            loss.backward();
            model->optimizer()->step();
    
            //std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "] Loss: " << loss.item<float>() << std::endl;
        }
    } catch (const std::exception& e) {
	std::cerr << "Exception - " << e.what() << std::endl ;
    }
}
