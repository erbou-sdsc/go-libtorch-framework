#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <torch/torch.h>

class device_manager {
public:
    static void device(const std::string_view device_str);
    static torch::Device device();

private:
    device_manager() = default;
    void init();
    torch::Device device_{torch::kCPU};
    std::once_flag once_;
    static device_manager& instance();
};

class Tensor {
public:
    Tensor(void* blob, at::IntArrayRef shape, torch::TensorOptions& options) :
        tensor_{torch::from_blob(blob, shape, options).clone()} {}
    Tensor(torch::Tensor&& tensor) : tensor_{tensor} {}
    Tensor(Tensor const&) = delete;
    Tensor() = delete;

    operator torch::Tensor& () {
        return tensor_ ;
    }

    operator torch::Tensor const& () const {
        return tensor_ ;
    }

    void to(torch::Device const& device) {
        tensor_ = tensor_.to(device);
    }

    torch::Tensor const& tensor() const {
        return tensor_ ;
    }

private:
    torch::Tensor tensor_ ;
};


// struct Model : torch::nn::Module {
struct Model {
    using shape_t = at::IntArrayRef ;

    Model() : device{device_manager::device()} {}
    virtual ~Model() = default ;

    virtual torch::Tensor forward(torch::Tensor const x) = 0 ;
    virtual std::shared_ptr<torch::optim::Optimizer> optimizer() = 0 ;
    virtual torch::nn::Module& module() = 0 ;
    virtual void load_weights(const std::vector<char>& buffer) = 0 ;
    virtual torch::Tensor prepare_targets(torch::Tensor const& output, std::vector<torch::Tensor const*> targets) = 0 ;

    torch::nn::Module const& module() const {
        return module();
    }

    operator torch::nn::Module& () {
        return module();
    }

    operator torch::nn::Module const& () const {
        return module();
    }

    torch::Device const device;
};

class model_factory {
public:
    using model_constructor = std::function<Model*(std::string_view)>;

    static void registerModel(std::string&& name, model_constructor cnstr);
    static Model* newModel(const std::string& model, std::string_view opt = {});

private:
    model_factory() = default;
    ~model_factory();


    static model_factory& instance();
    void load_model_plugins(const std::string_view& directory);
    void init();

    std::mutex mtx_;
    std::once_flag once_;

    std::unordered_map<std::string, model_constructor> registry_;
    std::vector<void*> plugin_handles;

    // disallow copy/move
    model_factory(const model_factory&) = delete;
    model_factory& operator=(const model_factory&) = delete;
};
