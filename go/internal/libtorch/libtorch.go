package main

//! See conditions for noescape and nocallback optimization

/*
#cgo LDFLAGS: -L. -ltest_cnn
#include <stdlib.h>
#include "torch_bind.h"
#cgo noescape NewModel
#cgo noescape NewTensor
#cgo noescape FreeModel
#cgo noescape FreeTensor
#cgo noescape Infer
#cgo noescape Train
#cgo nocallback NewModel
#cgo nocallback NewTensor
#cgo nocallback FreeModel
#cgo nocallback FreeTensor
#cgo nocallback Infer
#cgo nocallback Train
*/
import "C"

import (
    "unsafe"

    "os"
    "fmt"
    "math/rand"
    "sync"
    "strconv"
    "time"
)

// Go Bindings to C functions in test_cnn.cpp
// {
func NewTensor(data []float32) *C.struct_tensor {
}

func FreeTensor(tensor *C.struct_tensor) {
}

func (model *C.struct_CNN) DeleteModel() {
    C.DeleteModel(model)
}

func NewModelWithOpts (model string, options string) *C.struct_basic_model {
    cModel := C.CString(model)
    cOpts := C.CString(options)
    defer C.free(unsafe.Pointer(cModel))
    defer C.free(unsafe.Pointer(cOpts))
    return C.NewModel(cModel, cOpts)
}

func NewModel(model string) *C.struct_basic_model {
    return NewModelWithOpts(model, "")
}

func (model *C.struct_basic_model) FreeModel() {
    C.DeleteModel(model)
}

func Train(model *C.struct_basic_model, tensor *C.struct_tensor, target tensor *C.struct_tensor, num_epochs int) {
    cEpochs := (C.int)(num_epochs)
    C.torch_training(model, tensor, target, cEpochs)
}

func InferenceModel(model *C.struct_basic_model, tensor *C.struct_tensor, result []float32) int {
    cResult := (*C.float)(unsafe.Pointer(&result[0]))
    cMaxSize := (C.size_t)(len(result))
    return int(C.torch_inference(model, tensor, cResult, cMaxSize))
}
// }

type Message struct {
    Input interface{}
    Label interface{}
    Callback chan<- interface{}
}

func InferenceAggregator(model *C.struct_CNN, aggregatorChannel <-chan Message, batchSize int, timeoutMs time.Duration) {
    outputSize   := int(C.torch_model_output_size(model))
    inputSize    := int(C.torch_model_input_size(model))
    batchData    := make([]float32, 0, batchSize * inputSize)
    batchChan    := make([]chan<- []float32, 0, batchSize)
    resultBuffer := make([]float32, batchSize * outputSize)

    for true {
        timeout := false
        select {
            case request := <- aggregatorChannel:
                batchData = append(batchData, request.Data...)
                batchChan = append(batchChan, request.Callback)
            case <-time.After(timeoutMs * time.Millisecond):
	        timeout = true
                fmt.Printf(`O`)
        }
        if (len(batchChan) >= batchSize || timeout) && len(batchData) > 0 {
            fmt.Printf(`*`)
            resultSize := InferenceModel(model, batchData, resultBuffer)
            fmt.Printf(`+`)
            if resultSize > 0 {
                for i, callback := range batchChan {
                    callback <- resultBuffer[i*outputSize : (i+1)*outputSize]
                }
            }
            batchData = batchData[:0]
            batchChan = batchChan[:0]
        }
    }
}

