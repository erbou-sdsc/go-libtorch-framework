package ml

/*
#cgo CXXFLAGS: -std=c++20
#include <stdlib.h>
#include "torch_bind.h"
*/
import "C"

import (
	//"unsafe"
	//"os"
	"fmt"
	//"math/rand"
	//"sync"
	//"strconv"
	"time"
)

type InferenceInput[T any, K any] struct {
	Data     []T
	Callback chan<- []K
}

func InferenceAggregator[T any, K any](model Model, aggregatorChannel <-chan InferenceInput[T, K], batchSize int, timeoutMs time.Duration) {
	batchData := make([]T, 0, batchSize)
	batchChan := make([]chan<- []K, 0, batchSize)

	for true {
		timeout := false
		select {
		case request := <-aggregatorChannel:
			batchData = append(batchData, request.Data...)
			batchChan = append(batchChan, request.Callback)
		case <-time.After(timeoutMs * time.Millisecond):
			timeout = true
			fmt.Printf(`O`)
		}
		if (len(batchChan) >= batchSize || timeout) && len(batchData) > 0 {
			fmt.Printf(`*`)
			var result Tensor
			if flattened, err := Flatten[T](batchData); err == nil {
				tensor := CreateTensor(flattened)
				if tensor.IsValid() {
					result = model.Infer(tensor)
				}
				tensor.Delete()
			}
			if result.IsValid() {
				fmt.Printf(`+`)
				nShape := result.Shape()
				if nShape[0] == len(batchChan) {
					// TODO split result batch into individual results
				}
				for i, callback := range batchChan {
					if i > 0 {
					}
					// TODO
					//callback <- result
					callback <- nil
				}
			} else {
				// No result returned, or there was an error.
				// Returns nil to immediately notify the agent so that they stop waiting for the result.
				fmt.Print(`!`)
				for _, callback := range batchChan {
					callback <- nil
				}
			}
			batchData = batchData[:0]
			batchChan = batchChan[:0]
		}
	}
}

/*
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
*/
