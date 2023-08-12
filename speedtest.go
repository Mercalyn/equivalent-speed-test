package main

/*
speed test
qty 2 prefilled array size [12k, 12k] multiplying into the first,
10 iterations
consistent result:
reg loop: 4.5 sec
goroutines: 940 ms
i think go is natively using multi cpu cores making it quick by default, whereas my c/c++ program do not
*/

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

const (
	numLoops   int = 10
	sizeXArray int = 12000
	sizeYArray int = 12000
)

func process(aArr *[][]float64, bArr *[][]float64, wg *(sync.WaitGroup)) {
	// inside goroutine, loop thru length of this
	for y := 0; y < len(*aArr); y++ {
		//fmt.Println((*arr)[y])
		for x := 0; x < len((*aArr)[y]); x++ {
			(*aArr)[y][x] *= (*bArr)[y][x]
			//fmt.Println((*aArr)[y][x])
		}
	}
	//fmt.Println((*aArr)[0][0]);
	wg.Done()
}

func postAdd(aArr *[][]float64) {
	// inside goroutine, loop thru length of this
	for y := 0; y < len(*aArr); y++ {
		//fmt.Println((*arr)[y])
		for x := 0; x < len((*aArr)[y]); x++ {
			(*aArr)[y][x] += 1.2
			//fmt.Println((*arr)[y][x])
		}
	}
}

func prefill(arr *[][]float64, wg *(sync.WaitGroup)){
	for y := 0; y < len(*arr); y++ {
		//fmt.Println((*arr)[y])
		for x := 0; x < len((*arr)[y]); x++ {
			(*arr)[y][x] = (rand.Float64() * 2) + .0001
			//fmt.Println((*arr)[y][x])
		}
	}
	wg.Done()
}

func main() {
	// start wait group as a pointer
	wg := new(sync.WaitGroup)
	
	// make 2 2d arrays "slices"
	//processArray := make([]float64, sizeXArray)
	aArr := make([][]float64, (sizeYArray))
	bArr := make([][]float64, (sizeYArray))
	for y := 0; y < sizeYArray; y++ {
		aArr[y] = make([]float64, sizeXArray)
		bArr[y] = make([]float64, sizeXArray)
	}
	
	// prefill
	wg.Add(2)
	go prefill(&aArr, wg)
	go prefill(&bArr, wg)
	wg.Wait()
	fmt.Println("done with prefill");
	/*
	fmt.Println(aArr[0][0]);
	fmt.Println(bArr[0][0]);
	*/


	// start timer, process
	timerNow := time.Now()
	timerStart := (timerNow.Second() * 1e9) + timerNow.Nanosecond()
	for i := 0; i < numLoops; i++ {
		wg.Add(1)
		go process(&aArr, &bArr, wg)
	}

	// wait on waitgroup then stop timer
	wg.Wait()
	timerNow = time.Now()
	timerEnd := (timerNow.Second() * 1e9) + timerNow.Nanosecond()
	timerFinal := (timerEnd - timerStart) / 1000000
	//fmt.Println(aArr[11000][11000])
	fmt.Println(timerFinal, "ms")
	
	// added a post timer function to make sure it isn't cheating
	postAdd(&aArr)
	postAdd(&bArr)
	//fmt.Println(aArr[11000][11000])
}
