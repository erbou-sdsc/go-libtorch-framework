package main

//! See conditions for noescape and nocallback optimization

/*
#cgo LDFLAGS: -L. -ltest_cnn
#include <stdlib.h>
#include "test_cnn.h"
#cgo noescape torch_training
#cgo noescape torch_inference
#cgo noescape torch_model_output_size
#cgo noescape torch_model_input_size
#cgo nocallback torch_training
#cgo nocallback torch_inference
#cgo nocallback torch_model_output_size
#cgo nocallback torch_model_input_size
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
func InitializeModelWithOpts (model string, options string) *C.struct_CNN {
    cModel := C.CString(model)
    cOpts := C.CString(options)
    defer C.free(unsafe.Pointer(cModel))
    defer C.free(unsafe.Pointer(cOpts))
    return C.torch_initialize_model(cModel, cOpts)
}

func InitializeModel(model string) *C.struct_CNN {
    return InitializeModelWithOpts(model, "")
}

func DeleteModel(model *C.struct_CNN) {
    C.torch_delete_model(model)
}

func TrainModel(model *C.struct_CNN, data []float32, target []int, num_epochs int) {
    cData := (*C.float)(unsafe.Pointer(&data[0]))
    cTarget := (*C.int)(unsafe.Pointer(&target[0]))
    cSize := (C.size_t)(len(data))
    cEpochs := (C.int)(num_epochs)
    C.torch_training(model, cData, cTarget, cSize, cEpochs)
}

func InferenceModel(model *C.struct_CNN, data []float32, result []float32) int {
    cData := (*C.float)(unsafe.Pointer(&data[0]))
    cSize := (C.size_t)(len(data))
    cResult := (*C.float)(unsafe.Pointer(&result[0]))
    cMaxSize := (C.size_t)(len(result))
    return int(C.torch_inference(model, cData, cSize, cResult, cMaxSize))
}
// }

type InferenceInput struct {
    Data []float32
    Callback chan<- []float32
}

func InferenceAggregator(model *C.struct_CNN, aggregatorChannel <-chan InferenceInput, batchSize int, timeoutMs time.Duration) {
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

func simGenerateRandomData(id int, aggregatorChannel chan<- InferenceInput, numRequests int, dataSize int, wg *sync.WaitGroup) {
    callback := make(chan []float32, 1)
    defer close(callback)
    defer wg.Done()

    for range numRequests {
        data := make([]float32, dataSize)
        for i := 0; i < dataSize; i++ {
            data[i] = rand.Float32()
        }
        fmt.Print(`:`)
        aggregatorChannel <- InferenceInput{ data, callback }
        select {
            case result := <- callback:
                if (len(result) == 0) {
                    fmt.Printf(`(%v)`, id)
                } else {
                    fmt.Print(`|`)
                }
            case <-time.After(5 * time.Second):
                fmt.Print(`?`)
                return
        }
    }
    fmt.Print(`.`)
}

func main() {
    batchSize := 100
    numSenders := 2*batchSize
    numRequests := 5 // # request per sender
    timeoutMs := 500 // # Aggregator sends partial batch after waiting that many  ms

    if len(os.Args)>1 {
        switch os.Args[1] {
            case "-h", "--help", "help":
                fmt.Println("usage: batchSize numSenders requestsPerSender [timeoutMs]")
                return
            default:
                atoi := func (s string) int {
                     if i, err := strconv.Atoi(s); err == nil {
                         return i
                     } else {
                         panic(fmt.Sprintf("error: expected int, got '%v'\n", s))
                     }
                }
                if len(os.Args) > 3 {
                    batchSize = atoi(os.Args[1])
                    numSenders = atoi(os.Args[2])
                    numRequests = atoi(os.Args[3])
                }
                if len(os.Args) > 4 {
                    timeoutMs = atoi(os.Args[4])
                }
        }
    }

    model := InitializeModel("CNN_Mnist")
    defer DeleteModel(model)
    if model == nil {
        fmt.Println("No model")
        return
    }

    inputSize := int(C.torch_model_input_size(model))
    aggregatorQueueSize := numSenders

    fmt.Printf("Request aggregator, batch size: %v, input size: %v (tot: %v)\n", batchSize, inputSize, batchSize * inputSize)
    fmt.Printf("Receives requests from %v random data generator goroutines, %v requests each\n\n", numSenders, numRequests)

    data := make([]float32, batchSize * inputSize)
    target := make([]int, batchSize)
    TrainModel(model, data, target, 2)

    aggregatorChannel := make(chan InferenceInput, aggregatorQueueSize)
    defer close(aggregatorChannel)

    go InferenceAggregator(model, aggregatorChannel, batchSize, time.Duration(timeoutMs))

    var wg sync.WaitGroup
    wg.Add(numSenders)

    fmt.Println(`>>>>>>>>>> events <<<<<<<<<<`)
    fmt.Println(`:   A goroutine sent a request to the request aggregator`)
    fmt.Println(`*   The request aggregator forwards a batch of requests for inference on the model`)
    fmt.Println(`+   Request aggregator gets a batch of results from the model, and dispatched them to goroutines`)
    fmt.Println(`|   A goroutine received a result`)
    fmt.Println(`O   Request aggregator timed out while waiting for new input (submit partial batch)`)
    fmt.Println(`?   A goroutine timed-out and terminated while waiting for the result`)
    fmt.Println(`.   A goroutine gracefuly terminated after sending its requests and receiving all the responses.`)
    fmt.Println(`$   End`)
    fmt.Print("\n\n")

    for k := range numSenders {
        go simGenerateRandomData(k, aggregatorChannel, numRequests, inputSize, &wg)
    }

    wg.Wait()
    fmt.Println(`$`)
}
