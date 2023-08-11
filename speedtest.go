package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

/*
speed test comparison using go routine
*/

const (
	numLoops   int = 100000
	sizeXArray int = 16000
)

func process(xArray *[]float64, wg *(sync.WaitGroup)) {
	// inside goroutine, still loop thru length of this
	for i := 0; i < len(*xArray); i++ {
		//(*xArray)[i] = float32(i)
		//fmt.Println((*xArray)[i])

		(*xArray)[i] = ((1.78939423494 * 0.785186454) + (4.113810156 * 0.500005123)*rand.NormFloat64())
	}
	wg.Done()
}

func main() {
	// make array
	processArray := make([]float64, sizeXArray)

	// start wait group as a pointer
	wg := new(sync.WaitGroup)

	// start timer, process
	timerNow := time.Now()
	timerStart := (timerNow.Second() * 1e9) + timerNow.Nanosecond()
	for i := 0; i < numLoops; i++ {
		wg.Add(1)
		go process(&processArray, wg)
	}

	// wait on waitgroup then stop timer
	wg.Wait()
	timerNow = time.Now()
	timerEnd := (timerNow.Second() * 1e9) + timerNow.Nanosecond()
	timerFinal := (timerEnd - timerStart) / 1000000
	fmt.Println(timerFinal, "ms")

	fmt.Println(*&processArray[15000])
}
