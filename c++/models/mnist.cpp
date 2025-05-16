#include "../torch.h"

class Mnist : public Model {
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

extern "C" void RegisterModels() {
    std::cerr << "Register models mnist" << std::endl;
    model_factory::registerModel("mnist", [](std::string_view) {
        return new Mnist();
    });
}
