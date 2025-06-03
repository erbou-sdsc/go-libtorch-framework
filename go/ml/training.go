package ml

/*
#cgo CXXFLAGS: -std=c++20
#include <stdlib.h>
#include "torch_bind.h"
*/
import "C"
import "unsafe"

func trainModel(model Model, data Tensor, targets []*Tensor, num_epochs int) {
	cEpochs := (C.int)(num_epochs)
	cNumTargets := (C.int)(len(targets))
	cTargets := make([]*C.struct_Tensor, len(targets))
	for i, t := range targets {
		cTargets[i] = (*C.struct_Tensor)(t.p)
	}
	C.Train(
		model.p,
		data.p,
		(**C.struct_Tensor)(unsafe.Pointer(&cTargets[0])),
		cNumTargets,
		cEpochs)
}
