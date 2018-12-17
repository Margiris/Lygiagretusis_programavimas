package main

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
	"time"
)

const (
	length int = 1000000
)

func compareAndSwap(a []int, i, j int, ascending bool) {
	if ascending == (a[i] > a[j]) {
		a[i], a[j] = a[j], a[i]
	}
}

func greatestPowerOfTwoLessThan(n int) int {
	var k = 1
	
	for k > 0 && k < n {
		k = k << 1
	}
	
	return k >> 1
}

func bitonicMerge(a []int, low, length int, ascending bool, guardChan chan struct{}) {
	if length > 1 {
		var k = greatestPowerOfTwoLessThan(length)
		
		for i := low; i < low+length-k; i++ {
			compareAndSwap(a, i, i+k, ascending)
		}
		
		var chan1 = make(chan struct{})
		var chan2 = make(chan struct{})
		
		go bitonicMerge(a, low, k, ascending, chan1)
		go bitonicMerge(a, low+k, length-k, ascending, chan2)
		
		<-chan1
		<-chan2
	}
	
	guardChan <- struct{}{}
}

func bitonicSort(a []int, low, length int, ascending bool, guardChan chan struct{}) {
	if length > 1 {
		var k = length / 2
		
		var chan1 = make(chan struct{})
		var chan2 = make(chan struct{})
		
		go bitonicSort(a, low, k, ascending, chan1)
		go bitonicSort(a, low+k, length-k, !ascending, chan2)
		
		<-chan1
		<-chan2
		
		bitonicMerge(a, low, length, ascending, guardChan)
	} else {
		guardChan <- struct{}{}
	}
}

func sort(a []int, ascending bool) {
	var guardChan = make(chan struct{})
	go bitonicSort(a, 0, len(a), ascending, guardChan)
	<-guardChan
}

func printArray(a []int) {
	var buffer strings.Builder
	
	for _, number := range a {
		buffer.WriteString(fmt.Sprintf("%d ", number))
	}
	
	fmt.Println(buffer.String())
}

func generateRandomIntegerArray(size int) []int {
	arr := make([]int, size)
	
	rand.Seed(time.Now().Unix())
	
	for i := 0; i < size; i++ {
		arr[i] = rand.Intn(size * 3)
		
		if rand.Intn(2) > 0 {
			arr[i] = 0 - arr[i]
		}
	}
	
	return arr
}

func main() {
	
	var a0 = generateRandomIntegerArray(length)
	var a1 = generateRandomIntegerArray(length)
	
	// printArray(a0)
	// printArray(a1)
	
	sort(a0, false)
	sort(a1, true)
	
	fmt.Println("")
	
	// printArray(a0)
	// printArray(a1)
}

func BenchmarkBitonicSort(b *testing.B){
	var array = generateRandomIntegerArray(length)
	
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		sort(array, true)
	}
}
