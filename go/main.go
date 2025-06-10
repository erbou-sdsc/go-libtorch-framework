package main

import (
	"fmt"
	"math/rand/v2"
	"os"
	"strconv"
	"sync"
	"time"

	ml "goTorch/ml"
)

func simGenerateRandomData3Dx1D(id int, aggregatorChannel chan<- ml.InferenceInput[[][][]float32, []float32], numRequests int, dataSize int, wg *sync.WaitGroup) {
	callback := make(chan []float32, 1)
	defer close(callback)
	defer wg.Done()

	for range numRequests {
		data := make([][][]float32, 1)
		data[0] = make([][]float32, dataSize)
		for i := 0; i < dataSize; i++ {
			data[0][i] = make([]float32, dataSize)
			for j := 0; j < dataSize; j++ {
				data[0][i][j] = rand.Float32()
			}
		}
		fmt.Print(`:`)
		aggregatorChannel <- ml.InferenceInput[[][][]float32, []float32]{data, callback}
		select {
		case result := <-callback:
			if len(result) == 0 {
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
	numSenders := 2 * batchSize
	numRequests := 5 // # request per sender
	timeoutMs := 500 // # Aggregator sends partial batch after waiting that many  ms

	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "-h", "--help", "help":
			fmt.Println("usage: batchSize numSenders requestsPerSender [timeoutMs]")
			return
		default:
			atoi := func(s string) int {
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

	fmt.Printf("batchSize: %v, numSenders: %v, requestsPerSender: %v, timeoutMs: %v\n", batchSize, numSenders, numRequests, timeoutMs)

	model := ml.CreateModel("mnist")
	defer model.Delete()
	if !model.IsValid() {
		fmt.Println("No model")
		return
	}

	rows, cols, depth := 3, 4, 2
	data := make([]float64, cols*rows*depth)
	shape := []int{rows, cols, depth}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			for z := 0; z < depth; z++ {
				data[i*cols*depth+j*depth+z] = (float64)((float32)(i)*1.4 + (float32)(j)*1.6 + (float32)(z)*1.8)
			}
		}
	}

	tensor := ml.CreateTensor(ml.Flattened[float64]{Data: data, Shape: shape})
	defer tensor.Delete()
	if !tensor.IsValid() {
		fmt.Println("Bad Tensor")
		return
	}
	fmt.Println(tensor.DType())
	fmt.Println(tensor.Shape())
	fmt.Println(tensor.Bytes())
	fmt.Println(data)
	fmt.Println(tensor.Data())
	tensor.Delete()

	/*
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
	*/
}
