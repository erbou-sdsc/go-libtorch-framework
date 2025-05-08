#pragma once

#include <functional>
#include <string>
#include <torch/torch.h>

class device_manager {
public:
    static void device(const std::string_view device_str);
    static torch::Device device();

private:
    device_manager();
    torch::Device device_{torch::kCPU};
    static device_manager& instance();
    static std::mutex& mutex();
};

struct basic_model : torch::nn::Module {
    using shape_t = at::IntArrayRef ;

    basic_model() : device{device_manager::device()} {}
    virtual ~basic_model() = default ;

    virtual torch::Tensor forward(torch::Tensor const x) = 0 ;
    virtual const shape_t ishape() const = 0 ;
    virtual const shape_t oshape() const = 0 ;
    virtual const torch::ScalarType itype() const = 0 ;
    virtual const torch::ScalarType otype() const = 0 ;
    virtual std::shared_ptr<torch::optim::Optimizer> optimizer() = 0 ;

    const size_t isize() const {
        return [](shape_t s) { return std::accumulate(s.begin(), s.end(), 1, std::multiplies()); }(ishape());
    }

    const size_t osize() const {
        return [](auto s) { return std::accumulate(s.begin(), s.end(), 1, std::multiplies()); }(oshape());
    }

    torch::Device const device;
};

class model_factory {
public:
    using model_constructor = std::function<basic_model*(std::string_view)>;

    static void registerModel(std::string&& name, model_constructor cnstr);
    static basic_model* newModel(const std::string& model, std::string_view opt = {});
    void load_model_plugins(const std::string_view& directory);

private:
    model_factory();
    ~model_factory();

    static model_factory& instance();
    static std::mutex& mutex();

    std::unordered_map<std::string, model_constructor> registry_;
    std::vector<void*> plugin_handles;

    // disallow copy/move
    model_factory(const model_factory&) = delete;
    model_factory& operator=(const model_factory&) = delete;
};
