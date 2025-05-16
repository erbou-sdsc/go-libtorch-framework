package ml

/*
#cgo CXXFLAGS: -std=c++20
#include <stdlib.h>
#include "torch_bind.h"
*/
import "C"

func TrainModel(model Model, data Tensor, target Tensor, num_epochs int) {
    cEpochs := (C.int)(num_epochs)
    C.Train(model.p, data.p, target.p, cEpochs)
}

