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
#cgo noescape   Shape
#cgo noescape   Dim
#cgo noescape   NumBytes
#cgo noescape   DType
#cgo noescape   NumElem
#cgo noescape   ElemSize
#cgo noescape   Bytes
#cgo noescape   FreeTensor
#cgo nocallback NewModel
#cgo nocallback FreeModel
#cgo nocallback NewTensorInt8
#cgo nocallback NewTensorInt16
#cgo nocallback NewTensorInt32
#cgo nocallback NewTensorInt64
#cgo nocallback NewTensorUInt8
#cgo nocallback NewTensorUInt16
#cgo nocallback NewTensorUInt32
#cgo nocallback NewTensorUInt64
#cgo nocallback NewTensorFloat32
#cgo nocallback NewTensorFloat64
#cgo nocallback Shape
#cgo nocallback Dim
#cgo nocallback NumBytes
#cgo nocallback DType
#cgo nocallback NumElem
#cgo nocallback ElemSize
#cgo nocallback Bytes
#cgo nocallback FreeTensor
*/
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"reflect"
	"runtime"
	"sync"
	"unsafe"
)

// Handle to libtorch Model
type Model struct {
	p    *C.struct_Model
	once sync.Once
}

// Handle to libtorch Tensor
type Tensor struct {
	p    *C.struct_Tensor
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
func CreateModelWithOpts(model string, options string) Model {
	cModel := C.CString(model)
	cOpts := C.CString(options)
	defer C.free(unsafe.Pointer(cModel))
	defer C.free(unsafe.Pointer(cOpts))
	o := Model{p: C.NewModel(cModel, cOpts)}
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
func (o Model) Delete() {
	o.once.Do(func() {
		if o.p != nil {
			C.FreeModel(o.p)
			o.p = nil
		}
	})
}

// IsValid reports whether the model is properly initialized and usable.
func (o Model) IsValid() bool {
	return o.p != nil
}

func (o Model) Infer(data Tensor) Tensor {
	return Tensor{p: C.Infer(o.p, data.p)}
}

func (o Model) Train(data Tensor, targets []*Tensor, epochs int) {
	trainModel(o, data, targets, epochs)
}

// CreateTensor creates a new Tensor from a flattened data slice and shape.
// If creation fails, the returned Tensor will report IsValid() == false.
func CreateTensor[T any](flattened Flattened[T]) Tensor {
	var o Tensor

	if len(flattened.Data) == 0 {
		return Tensor{p: nil}
	}

	cData := unsafe.Pointer(&flattened.Data[0])
	cShape := (*C.size_t)(unsafe.Pointer(&flattened.Shape[0]))
	cDim := (C.size_t)(len(flattened.Shape))

	switch any(flattened.Data).(type) {
	case []uint8:
		o = Tensor{p: C.NewTensorUInt8(cData, cShape, cDim)}
	case []uint16:
		o = Tensor{p: C.NewTensorUInt16(cData, cShape, cDim)}
	case []uint32:
		o = Tensor{p: C.NewTensorUInt32(cData, cShape, cDim)}
	case []uint64:
		o = Tensor{p: C.NewTensorUInt64(cData, cShape, cDim)}
	case []int8:
		o = Tensor{p: C.NewTensorInt8(cData, cShape, cDim)}
	case []int16:
		o = Tensor{p: C.NewTensorInt16(cData, cShape, cDim)}
	case []int32:
		o = Tensor{p: C.NewTensorInt32(cData, cShape, cDim)}
	case []int64:
		o = Tensor{p: C.NewTensorInt64(cData, cShape, cDim)}
	case []float32:
		o = Tensor{p: C.NewTensorFloat32(cData, cShape, cDim)}
	case []float64:
		o = Tensor{p: C.NewTensorFloat64(cData, cShape, cDim)}
	default:
		return Tensor{p: nil}
	}

	runtime.SetFinalizer(&o, func(t *Tensor) {
		t.Delete()
	})

	return o
}

func (t Tensor) Dim() int {
	return int(C.Dim((*C.struct_Tensor)(t.p)))
}

func (t Tensor) DType() string {
	return C.GoString(C.DType(t.p))
}

func (t Tensor) Shape() []int {
	ndim := t.Dim()
	ptr := C.Shape((*C.struct_Tensor)(t.p))
	if ptr == nil || ndim <= 0 {
		return nil
	}

	// Create Go slice that views the C array temporarily
	shapeView := unsafe.Slice((*int64)(unsafe.Pointer(ptr)), int(ndim))

	// Copy to Go-managed slice
	shape := make([]int, int(ndim))
	for i, v := range shapeView {
		shape[i] = int(v)
	}

	return shape
}

func (t Tensor) NumBytes() int {
	return int(C.NumBytes(t.p))
}

func (t Tensor) NumElem() int {
	return int(C.NumElem(t.p))
}

func (t Tensor) ElemSize() int {
	return int(C.ElemSize(t.p))
}

func (t Tensor) Bytes() []byte {
	var bufferSize C.size_t = C.NumBytes(t.p)
	buf := make([]byte, bufferSize)
	copied := C.Bytes(t.p, unsafe.Pointer(&buf[0]), bufferSize)
	if copied != bufferSize {
		return nil
	}
	return buf
}

func deflatRecursive[T any](data []byte, shape []int, size int) (any, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("Invalid shape")
	}

	if len(shape) == 1 {
		if len(data)%size != 0 {
			return nil, fmt.Errorf("data size %d is not a multiple of element size %d", len(data), size)
		}
		count := len(data) / size
		result := make([]T, count)
		buf := bytes.NewReader(data)
		if err := binary.Read(buf, binary.NativeEndian, result); err != nil {
			return nil, err
		}
		return result, nil
	}

	array := make([]any, shape[0])
	sliceByteSize := len(data) / shape[0]
	if sliceByteSize*shape[0] != len(data) {
		return nil, fmt.Errorf("Jagged tensor")
	}

	for i := 0; i < shape[0]; i++ {
		sub, err := deflatRecursive[T](data[i*sliceByteSize:(i+1)*sliceByteSize], shape[1:], size)
		if err != nil {
			return nil, err
		}
		array[i] = sub
	}

	return array, nil
}

func deflatRecursiveReflect[T any](data []byte, shape []int, size int) (any, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("invalid shape")
	}

	if len(shape) == 1 {
		if len(data)%size != 0 {
			return nil, fmt.Errorf("data size %d is not multiple of element size %d", len(data), size)
		}
		count := len(data) / size
		result := make([]T, count)
		buf := bytes.NewReader(data)
		if err := binary.Read(buf, binary.NativeEndian, result); err != nil {
			return nil, err
		}
		return result, nil
	}

	arrayLen := shape[0]
	sliceByteSize := len(data) / arrayLen
	if sliceByteSize*arrayLen != len(data) {
		return nil, fmt.Errorf("jagged tensor")
	}

	// Create a slice type of nested slice for the remaining dimensions
	subSlice, err := deflatRecursiveReflect[T](data[:sliceByteSize], shape[1:], size)
	if err != nil {
		return nil, err
	}
	subSliceVal := reflect.ValueOf(subSlice)
	subSliceType := subSliceVal.Type()

	// Create the outer slice with length shape[0]
	outSlice := reflect.MakeSlice(reflect.SliceOf(subSliceType), arrayLen, arrayLen)

	for i := 0; i < arrayLen; i++ {
		sub, err := deflatRecursiveReflect[T](data[i*sliceByteSize:(i+1)*sliceByteSize], shape[1:], size)
		if err != nil {
			return nil, err
		}
		outSlice.Index(i).Set(reflect.ValueOf(sub))
	}

	return outSlice.Interface(), nil
}

func deflatTensor(data []byte, shape []int, dtype string) (any, error) {
	switch dtype {
	case "f32":
		return deflatRecursiveReflect[float32](data, shape, 4)
	case "f64":
		return deflatRecursiveReflect[float64](data, shape, 8)
	case "i8":
		return deflatRecursiveReflect[int8](data, shape, 1)
	case "i16":
		return deflatRecursiveReflect[int16](data, shape, 2)
	case "i32":
		return deflatRecursiveReflect[int32](data, shape, 4)
	case "i64":
		return deflatRecursiveReflect[int64](data, shape, 8)
	case "u8":
		return deflatRecursiveReflect[uint8](data, shape, 1)
	case "u16":
		return deflatRecursiveReflect[uint16](data, shape, 2)
	case "u32":
		return deflatRecursiveReflect[uint32](data, shape, 4)
	case "u64":
		return deflatRecursiveReflect[uint64](data, shape, 8)
	default:
		return nil, fmt.Errorf("unsupported dtype: %s", dtype)
	}
}

func (t Tensor) Data() (any, error) {
	return deflatTensor(t.Bytes(), t.Shape(), t.DType())
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
	equalShape := func(a, b []int) bool {
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
