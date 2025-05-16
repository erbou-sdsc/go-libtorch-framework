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

    "fmt"
    "reflect"
    "runtime"
    "sync"
    "unsafe"
)

// Handle to libtorch Model
type Model struct {
    p *C.struct_Model
    once sync.Once
}

// Handle to libtorch Tensor
type Tensor struct {
    p *C.struct_Tensor
    once sync.Once
}

// Flattened represents a tensor in Go.
// Data contains the elements flattened into a single-dimensional slice,
// and Shape holds the size of each dimension of the original tensor.
type Flattened[T any] struct {
    Data  []T
    Shape []int
}

// CreateModelWithOpts creates a new Model.
// The options string is model-specific and may vary depending on the implementation.
// If the model cannot be created successfully, the returned Model will report IsValid() == false.
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

// CreateModel creates a new Model with default values.
// If the model cannot be created successfully, the returned Model will report IsValid() == false.
func CreateModel(model string) Model {
    return CreateModelWithOpts(model, "")
}

// Delete releases native resources associated with the Model.
// Useful for freeing memory explicitly before garbage collection.
func (o *Model) Delete() {
    o.once.Do(func() {
        if o.p != nil {
            C.FreeModel(o.p)
            o.p = nil
        }
    })
}

// IsValid reports whether the model is properly initialized and usable.
func (o *Model) IsValid() bool {
    return o.p != nil
}

func (o *Model) Inference(data Tensor) Tensor {
    return Tensor{p:C.Infer(o.p, data.p)}
}

func (o *Model) Training(data Tensor, target Tensor, epochs int) {
    cEpochs  := (C.int)(epochs)
    C.Train(o.p, data.p, target.p, cEpochs)
}

// CreateTensor creates a new Tensor from a flattened data slice and shape.
// If creation fails, the returned Tensor will report IsValid() == false.
func CreateTensor[T any](flattened Flattened[T]) Tensor {
    var o Tensor

    if len(flattened.Data) == 0 {
        return Tensor{p:nil}
    }

    cData  := unsafe.Pointer(&flattened.Data[0])
    cSize  := (C.size_t)(len(flattened.Data))
    cShape := (*C.size_t)(unsafe.Pointer(&flattened.Shape[0]))
    cDim   := (C.size_t)(len(flattened.Shape))

    switch any(flattened.Data).(type) {
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

// Flatten takes an arbitrarily nested array (of arrays) of generic type T 
// and returns a Flattened[T] containing the flattened data and shape.
// The input can have any number of dimensions.
// Returns an error if input is not a valid nested array of T.
func Flatten[T any](input any) (Flattened[T], error) {
    var flat []T
    shape, err := flattenRecursive(reflect.ValueOf(input), &flat)
    if err != nil {
        return Flattened[T]{}, err
    }
    return Flattened[T]{Data: flat, Shape: shape}, nil
}

func flattenRecursive[T any](v reflect.Value, out *[]T) ([]int, error) {
    equalShape := func (a, b []int) bool {
        if len(a) != len(b) {
            return false
        }
        for i := range a {
            if a[i] != b[i] {
                return false
            }
        }
        return true
    }
    switch v.Kind() {
    case reflect.Slice, reflect.Array:
        length := v.Len()
        var innerShape []int
        for i := 0; i < length; i++ {
            subShape, err := flattenRecursive(v.Index(i), out)
            if err != nil {
                return nil, err
            }
            if i == 0 {
                innerShape = subShape
            } else if !equalShape(innerShape, subShape) {
                return nil, fmt.Errorf("ragged arrays not supported")
            }
        }
        return append([]int{length}, innerShape...), nil
    default:
        val := v.Interface()
        if t, ok := val.(T); ok {
            *out = append(*out, t)
            return nil, nil
        } else {
            return nil, fmt.Errorf("element type mismatch: expected %T, got %T", *new(T), val)
        }
    }
}

// IsValid reports whether the tensor is properly initialized and usable.
func (o *Tensor) IsValid() bool {
    return o.p != nil
}

// Delete releases native resources associated with the Tensor.
// Useful for freeing memory explicitly before garbage collection.
func (o *Tensor) Delete() {
    o.once.Do(func() {
        if o.p != nil {
            C.FreeTensor(o.p)
            o.p = nil
        }
    })
}

