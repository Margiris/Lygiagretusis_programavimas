// Margiris Burakauskas, IFF-6/10
package main

import (
	"fmt"
	"sync"
)

const multiplyProcessesCount = 4
const factorialOfNumber = 20
const threadToReturnResultTo = 0

var synchronizer = sync.WaitGroup{}

func User(toMultiplyProcess []chan []float64, resultChan <-chan float64) {
	defer synchronizer.Done()
	var factorialArray [factorialOfNumber]float64

	for i := 0; i < factorialOfNumber; i++ {
		factorialArray[i] = float64(i + 1)
	}

	for i := 0; i < multiplyProcessesCount; i++ {
		toMultiplyProcess[i] <- factorialArray[(i * 20 / multiplyProcessesCount) : (i+1)*20/multiplyProcessesCount]
	}

	var result = <-resultChan

	fmt.Printf("%.0d! = %.0f\n", factorialOfNumber, result)
}

func Multiply(threadIndex int, toMultiplyProcess <-chan []float64, toLastMultiplyProcess chan<- float64, resultChan chan<- float64) {
	defer synchronizer.Done()
	var factorialArraySlice = <-toMultiplyProcess

	var result = float64(1)

	for _, number := range factorialArraySlice {
		result *= number
	}

	toLastMultiplyProcess <- result

	if threadIndex == threadToReturnResultTo {
		var finalResult = <-toMultiplyProcess
		resultChan <- finalResult[0]
	}
}

func LastMultiply(toLastMultiplyProcess <-chan float64, toMultiplyProcess chan<- []float64) {
	defer synchronizer.Done()
	var result []float64

	result = append(result, float64(1))

	for i := 0; i < multiplyProcessesCount; i++ {
		result[0] *= <-toLastMultiplyProcess
	}

	toMultiplyProcess <- result
}

func main() {
	var toMultiplyProcess []chan []float64
	var toLastMultiplyProcess = make(chan float64)
	var resultChan = make(chan float64)

	for i := 0; i < multiplyProcessesCount; i++ {
		toMultiplyProcess = append(toMultiplyProcess, make(chan []float64))

		synchronizer.Add(1)
		go Multiply(i, toMultiplyProcess[i], toLastMultiplyProcess, resultChan)
	}

	synchronizer.Add(2)
	go User(toMultiplyProcess, resultChan)
	go LastMultiply(toLastMultiplyProcess, toMultiplyProcess[threadToReturnResultTo])

	synchronizer.Wait()
}
