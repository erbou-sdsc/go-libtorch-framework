#include "../torch.h"

class Mnist : public Model {
public:
    Mnist(std::string_view) {
        conv1 = _module.register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 5).stride(1).padding(2)));
        conv2 = _module.register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 5).stride(1).padding(2)));
        fc1 = _module.register_module("fc1", torch::nn::Linear(64 * 7 * 7, 1000));
        fc2 = _module.register_module("fc2", torch::nn::Linear(1000, 10));

        _module.to(device);

        // optimizer must be created after moving the model parameters to the device
        //_optimizer = std::make_shared<torch::optim::SGD>(_module.parameters(), torch::optim::SGDOptions(0.01));
        _optimizer = std::make_shared<torch::optim::Adam>(_module.parameters(), torch::optim::AdamOptions(1e-3));
    }

    torch::Tensor forward(torch::Tensor const x) override {
        auto t = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        t = torch::relu(torch::max_pool2d(conv2->forward(t), 2));
        t = t.view({-1, 64 * 7 * 7});
        t = torch::relu(fc1->forward(t));
        t = fc2->forward(t);
        return torch::log_softmax(t, 1);
    }

    torch::Tensor prepare_targets(torch::Tensor const& output, std::vector<torch::Tensor const*> targets) override {
        return torch::nll_loss(output, *targets[0]);
    }

    std::shared_ptr<torch::optim::Optimizer> optimizer() override {
        return _optimizer ;
    }

    torch::nn::Module& module () override {
        return _module ;
    }

    void load_weights(const std::vector<char>& buffer) override {
    }

private:
    torch::nn::Module _module;
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    const std::array<shape_t::value_type,3> i_shape{1,28,28};
    const shape_t::value_type o_shape{10};
    std::shared_ptr<torch::optim::Optimizer> _optimizer;
};

extern "C" void RegisterModels(model_factory& factory) {
    // std::cout << "Register models mnist" << std::endl;
    factory.registerModel("mnist", [](std::string_view opts) -> Model* {
        return new Mnist(opts);
    });
}
