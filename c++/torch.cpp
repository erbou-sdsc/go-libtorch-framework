#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <format>
#include <iostream>
#include <vector>
#include "torch.h"

void device_manager::device(const std::string_view device_str) {
    auto& inst = instance();
    inst.init();
    if (device_str == "cuda" && torch::cuda::is_available()) {
        inst.device_ = torch::Device(torch::kCUDA);
    } else if (device_str == "mps" && torch::mps::is_available()) {
        inst.device_ = torch::Device(torch::kMPS);
    } else if (device_str == "cpu") {
        inst.device_ = torch::Device(torch::kCPU);
    } else {
        throw std::runtime_error(std::format("Invalid or unavailable device: {}", device_str));
    }
}

torch::Device device_manager::device() {
    auto& inst = instance();
    inst.init();
    return inst.device_;
}

void device_manager::init() {
    std::call_once(once_, [this] {
        if (torch::cuda::is_available()) {
            device_ = torch::kCUDA;
            std::cout << "CUDA is available!" << std::endl;
        } else if (torch::mps::is_available()) {
            device_ = torch::kMPS;
            std::cout << "MPS is available!" << std::endl;
        } else {
            std::cout << "Using CPU" << std::endl;
        }
    });
}

device_manager& device_manager::instance() {
    static device_manager inst;
    return inst;
}

// model_factory::registerModel("mymodel", [](std::string_view opt) { return new MyModel(); });
void model_factory::registerModel(std::string&& name, model_constructor cnstr) {
    auto& inst = instance();
    inst.registry_[name] = cnstr ;
}

Model* model_factory::newModel(const std::string& model, std::string_view opt) {
    auto& inst = instance();
    inst.init();
    auto it = inst.registry_.find(model);
    if (it != inst.registry_.end()) {
        return it->second(opt);
    }
    throw std::runtime_error(std::format("Model not found - {}", model));
}

model_factory::~model_factory() {
    for (void* handle : plugin_handles) {
        dlclose(handle);
    }
}

void model_factory::load_model_plugins(const std::string_view& directory) {
    std::lock_guard<std::mutex> lock(mtx_);
    const char* model_path = std::getenv("LIBTORCH_MODEL_PATH");
    if (!model_path) {
        model_path = "./models";
    }
    for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
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

            plugin_handles.push_back(handle);
        }
    }
}

void model_factory::init() {
    std::call_once(once_, [this] {
        load_model_plugins("");
    });
}

model_factory& model_factory::instance() {
    static model_factory instance ;
    return instance ;
}
