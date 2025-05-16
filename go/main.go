package main

import (
    "fmt"
    "os"
    "strconv"

    ml "goTorch/ml"
)

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

    fmt.Printf("batchSize: %v, numSenders: %v, requestsPerSender: %v, timeoutMs: %v\n", batchSize, numSenders, numRequests, timeoutMs)

    model := ml.CreateModel("mnist")
    defer model.Delete()
    if model.Bad() {
        fmt.Println("No model")
        return
    }


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
