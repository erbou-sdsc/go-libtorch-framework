package ml

//! See conditions for noescape and nocallback optimization

/*
#cgo CXXFLAGS: -std=c++20
#include <stdlib.h>
#include "torch_bind.h"
#cgo noescape   NewModel
#cgo noescape   FreeModel
#cgo noescape   NewTensorInt8
#cgo noescape   NewTensorInt16
#cgo noescape   NewTensorInt32
#cgo noescape   NewTensorInt64
#cgo noescape   NewTensorUInt8
#cgo noescape   NewTensorUInt16
#cgo noescape   NewTensorUInt32
#cgo noescape   NewTensorUInt64
#cgo noescape   NewTensorFloat32
#cgo noescape   NewTensorFloat64
#cgo noescape   FreeTensor
#cgo nocallback NewModel
#cgo nocallback FreeModel
#cgo nocallback NewTensorUInt8
#cgo nocallback NewTensorUInt16
#cgo nocallback NewTensorUInt32
#cgo nocallback NewTensorUInt64
#cgo nocallback NewTensorFloat32
#cgo nocallback NewTensorFloat64
#cgo nocallback FreeTensor
*/
import "C"

import (
    "unsafe"

    "runtime"
    "sync"
)

type Model struct {
    p *C.struct_Model
    once sync.Once
}

type Tensor struct {
    p *C.struct_Tensor
    once sync.Once
}

func CreateModelWithOpts (model string, options string) Model {
    cModel := C.CString(model)
    cOpts := C.CString(options)
    defer C.free(unsafe.Pointer(cModel))
    defer C.free(unsafe.Pointer(cOpts))
    o := Model{p:C.NewModel(cModel, cOpts)}
    // Automatically delete when GC'd
    runtime.SetFinalizer(&o, func(m *Model) {
        m.Delete()
    })
    return o
}

func CreateModel(model string) Model {
    return CreateModelWithOpts(model, "")
}

func (o *Model) Delete() {
    o.once.Do(func() {
        if o.p != nil {
            C.FreeModel(o.p)
            o.p = nil
        }
    })
}

func (o *Model) Bad() bool {
    return o.p == nil
}

func (o Model) Inference(data Tensor) Tensor {
    //cResult := unsafe.Pointer(&result[0])
    //cMaxSize := (C.size_t)(len(result))
    //return int(C.Infer(o.p, data.p, cResult, cMaxSize))
    return Tensor{}
}

func CreateTensor[T any](data []T, shape []int) Tensor {
    var o Tensor

    if len(data) == 0 {
        return Tensor{p:nil}
    }

    cData  := unsafe.Pointer(&data[0])
    cSize  := (C.size_t)(len(data))
    cShape := (*C.size_t)(unsafe.Pointer(&shape[0]))
    cDim   := (C.size_t)(len(shape))

    switch any(data).(type) {
        case []uint8:
            o = Tensor{p:C.NewTensorUInt8(cData, cSize, cShape, cDim)}
        case []uint16:
            o = Tensor{p:C.NewTensorUInt16(cData, cSize, cShape, cDim)}
        case []uint32:
            o = Tensor{p:C.NewTensorUInt32(cData, cSize, cShape, cDim)}
        case []uint64:
            o = Tensor{p:C.NewTensorUInt64(cData, cSize, cShape, cDim)}
        case []int8:
            o = Tensor{p:C.NewTensorInt8(cData, cSize, cShape, cDim)}
        case []int16:
            o = Tensor{p:C.NewTensorInt16(cData, cSize, cShape, cDim)}
        case []int32:
            o = Tensor{p:C.NewTensorInt32(cData, cSize, cShape, cDim)}
        case []int64:
            o = Tensor{p:C.NewTensorInt64(cData, cSize, cShape, cDim)}
        case []float32:
            o = Tensor{p:C.NewTensorFloat32(cData, cSize, cShape, cDim)}
        case []float64:
            o = Tensor{p:C.NewTensorFloat64(cData, cSize, cShape, cDim)}
        default:
            return Tensor{p:nil}
    }

    runtime.SetFinalizer(&o, func(t *Tensor) {
        t.Delete()
    })

    return o
}

func (o *Tensor) Delete() {
    o.once.Do(func() {
        if o.p != nil {
            C.FreeTensor(o.p)
            o.p = nil
        }
    })
}

