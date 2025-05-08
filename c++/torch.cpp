#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <format>
#include <iostream>
#include <vector>
#include "torch.h"

// Example: MNIST Classifier Model
class Mnist : public basic_model {
public:
    Mnist() {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 5).stride(1).padding(2)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 5).stride(1).padding(2)));
        fc1 = register_module("fc1", torch::nn::Linear(64 * 7 * 7, 1000));
        fc2 = register_module("fc2", torch::nn::Linear(1000, 10));

        this->to(device);

        // optimizer must be created after moving the model parameters to the device
        //_optimizer = std::make_shared<torch::optim::SGD>(parameters(), torch::optim::SGDOptions(0.01));
        _optimizer = std::make_shared<torch::optim::Adam>(parameters(), torch::optim::AdamOptions(1e-3));
    }

    torch::Tensor forward(torch::Tensor const x) {
        auto t = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        t = torch::relu(torch::max_pool2d(conv2->forward(t), 2));
        t = t.view({-1, 64 * 7 * 7});
        t = torch::relu(fc1->forward(t));
        t = fc2->forward(t);
        return torch::log_softmax(t, 1);
    }

    std::shared_ptr<torch::optim::Optimizer> optimizer() {
        return _optimizer ;
    }

    const torch::ScalarType itype() const {
        return torch::kFloat ;
    }

    const torch::ScalarType otype() const {
        return torch::kFloat ;
    }

    const torch::ScalarType target_type() const {
        return torch::kInt ;
    }

    const shape_t ishape() const {
        return i_shape ;
    }

    const shape_t oshape() const {
        return o_shape ;
    }

private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    const std::array<shape_t::value_type,3> i_shape{1,28,28};
    const shape_t::value_type o_shape{10};
    std::shared_ptr<torch::optim::Optimizer> _optimizer;
};

void device_manager::device(const std::string_view device_str) {
    std::lock_guard<std::mutex> lock(mutex());
    if (device_str == "cuda" && torch::cuda::is_available()) {
        instance().device_ = torch::Device(torch::kCUDA);
    } else if (device_str == "mps" && torch::mps::is_available()) {
        instance().device_ = torch::Device(torch::kMPS);
    } else if (device_str == "cpu") {
        instance().device_ = torch::Device(torch::kCPU);
    } else {
        throw std::runtime_error(std::format("Invalid or unavailable device: {}", device_str));
    }
}

torch::Device device_manager::device() {
    return instance().device_;
}

device_manager::device_manager() {
    if (torch::cuda::is_available()) {
        device_ = torch::kCUDA;
        std::cout << "CUDA is available!" << std::endl;
    } else if (torch::mps::is_available()) {
        device_ = torch::kMPS;
        std::cout << "MPS is available!" << std::endl;
    } else {
        std::cout << "Using CPU" << std::endl;
    }
}

device_manager& device_manager::instance() {
    static device_manager instance;
    return instance;
}

std::mutex& device_manager::mutex() {
    static std::mutex mtx;
    return mtx;
}

model_factory::~model_factory() {
    for (void* handle : plugin_handles) {
        dlclose(handle);
    }
}

// model_factory::registerModel("mymodel", [](std::string_view opt) { return new MyModel(); });
void model_factory::registerModel(std::string&& name, model_constructor cnstr) {
    std::lock_guard<std::mutex> lock(mutex());
    instance().registry_[name] = cnstr ;
}

basic_model* model_factory::newModel(const std::string& model, std::string_view opt) {
    auto it = instance().registry_.find(model);
    if (it != instance().registry_.end()) {
        return it->second(opt);
    }
    throw std::runtime_error(std::format("Model not found - {}", model));
}

void model_factory::load_model_plugins(const std::string_view& directory) {
    std::lock_guard<std::mutex> lock(mutex());
    auto& self = instance();
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".so") {
            void* handle = dlopen(entry.path().c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (!handle) {
                std::cerr << "Failed to load " << entry.path() << ": " << dlerror() << std::endl;
                continue;
            }

            using RegisterFunc = void(*)();
            RegisterFunc register_models = reinterpret_cast<RegisterFunc>(dlsym(handle, "RegisterModels"));
            if (!register_models) {
                std::cerr << "RegisterModels not found in " << entry.path() << std::endl;
                dlclose(handle);
                continue;
            }

            register_models();

            self.plugin_handles.push_back(handle);
        }
    }
}

model_factory::model_factory() {
    const char* model_path = std::getenv("LIBTORCH_MODEL_PATH");
    if (!model_path) {
        model_path = "./models";
    }
    load_model_plugins(model_path);
}

model_factory& model_factory::instance() {
    static model_factory instance ;
    return instance ;
}

std::mutex& model_factory::mutex() {
    static std::mutex mtx;
    return mtx;
}

