package main

import (
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"sync"
	"time"
)

var size int
var threadCount int
var threadCountCurrent = 0
var mutex = &sync.Mutex{}

// grąžina, kiek laiko praėjo nuo start laiko
func timeTrack(start time.Time, name string) {
	elapsed := time.Since(start)
	fmt.Printf("%s took %s\n", name, elapsed)
}

// palygina i ir j elementus a masyve ir sukeičia vietomis jei neatitinka ascending reikšmės
func compareAndSwap(a []int, i, j int, ascending bool) {
	if ascending == (a[i] > a[j]) {
		a[i], a[j] = a[j], a[i]
	}
}

// func greatestPowerOfTwoLessThan(n int) int {
// 	var k = 1
//
// 	for k > 0 && k < n {
// 		k = k << 1
// 	}
//
// 	return k >> 1
// }
// func printArray(a []int) {
// 	var buffer strings.Builder
//
// 	for _, number := range a {
// 		buffer.WriteString(fmt.Sprintf("%d ", number))
// 	}
//
// 	fmt.Println(buffer.String())
// }

// surikiuoja ir sulieja dvi bitoninės eiles a masyve
func bitonicMerge(a []int, low, length int, ascending bool) {
	if length > 1 {
		var k = length / 2
		
		for i := low; i < low+k; i++ {
			compareAndSwap(a, i, i+k, ascending)
		}
		
		bitonicMerge(a, low, k, ascending)
		bitonicMerge(a, low+k, k, ascending)
	}
}

// surikiuoja ir sulieja dvi bitoninės eiles a masyve
// jei galima, visa tai daro lygiagrečiai
func bitonicMergeAsync(a []int, low, length int, ascending bool, guardChan chan<- struct{}, wasConcurrent bool) {
	if length > 1 {
		var k = length / 2
		
		for i := low; i < low+k; i++ {
			compareAndSwap(a, i, i+k, ascending)
		}
		
		mutex.Lock()
		if wasConcurrent && threadCountCurrent+2 <= threadCount {
			threadCountCurrent += 2
			mutex.Unlock()
			
			var chan1 = make(chan struct{})
			var chan2 = make(chan struct{})
			
			go bitonicMergeAsync(a, low, k, ascending, chan1, true)
			go bitonicMergeAsync(a, low+k, k, ascending, chan2, true)
			
			<-chan1
			<-chan2
		} else {
			mutex.Unlock()
			bitonicMerge(a, low, k, ascending)
			bitonicMerge(a, low+k, k, ascending)
		}
	}
	
	if wasConcurrent {
		guardChan <- struct{}{}
	}
}

// padalina a masyvą į dvi dalis ir joms iškviečia bitonicMerge
func bitonicSort(a []int, low, length int, ascending bool) {
	if length > 1 {
		var k = length / 2
		
		bitonicSort(a, low, k, ascending)
		bitonicSort(a, low+k, k, !ascending)
		
		bitonicMerge(a, low, length, ascending)
	}
}

// padalina a masyvą į dvi dalis ir joms iškviečia bitonicMerge
// jei galima, visa tai daro lygiagrečiai
func bitonicSortAsync(a []int, low, length int, ascending bool, guardChan chan<- struct{}, wasConcurrent bool) {
	if length > 1 {
		var k = length / 2
		
		mutex.Lock()
		if wasConcurrent && threadCountCurrent+2 <= threadCount {
			threadCountCurrent += 2
			mutex.Unlock()
			
			var chan1 = make(chan struct{})
			var chan2 = make(chan struct{})
			
			go bitonicSortAsync(a, low, k, ascending, chan1, true)
			go bitonicSortAsync(a, low+k, k, !ascending, chan2, true)
			
			<-chan1
			<-chan2
			
			bitonicMergeAsync(a, low, length, ascending, guardChan, true)
		} else {
			mutex.Unlock()
			bitonicSort(a, low, k, ascending)
			bitonicSort(a, low+k, k, !ascending)
			
			bitonicMerge(a, low, length, ascending)
		}
	}
	
	if wasConcurrent {
		guardChan <- struct{}{}
	}
}

// pagrindinė rikiavimo funkcija, pradeda rikiavimą ir matuoja bei į konsolę išveda vykdymo laiką
func sortB(a []int, ascending bool) {
	// kanalas, kuriuo gaunamas pranešimas apie rikiavimo pabaigą
	var guardChan = make(chan struct{})
	defer timeTrack(time.Now(), fmt.Sprintf("Bitonic sort of size %d using %d threads took", len(a), threadCount))
	
	go bitonicSortAsync(a, 0, len(a), ascending, guardChan, true)
	<-guardChan
	threadCountCurrent = 0
}

// grąžina size dydžio masyvą su pseudo-atsitiktinai sugeneruotais skaičiais
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
	// patikrinamas argumentų skaičius
	if len(os.Args) != 3 {
		fmt.Println("Netinkami argumentai. Naudojimas: [programos vardas].exe [gijų skaičius] [duomenų masyvo dydis]. Teisingam rikiavimui duomenų masyvo dydis turi būti dvejeto laipsnis.")
		return
	}
	
	// priskiriamos pradinės reikšmės iš argumentų
	var err error
	threadCount, err = strconv.Atoi(os.Args[1])
	size, err = strconv.Atoi(os.Args[2])
	
	// patikrinama, ar buvo pateikti teisingi argumentai
	if err != nil {
		fmt.Println("Netinkami argumentai. Naudojimas: [programos vardas].exe [gijų skaičius] [duomenų masyvo dydis]. Teisingam rikiavimui duomenų masyvo dydis turi būti dvejeto laipsnis.")
		return
	}
	
	// sukuriami duomenų masyvai
	var a = generateRandomIntegerArray(size)
	var a0 = make([]int, size)
	var a1 = make([]int, size)
	
	copy(a0, a)
	copy(a1, a)
	
	// paleidžiamas rikiavimas
	sortB(a0, true)
	sortB(a1, false)
	
	// patikrinama, ar buvo surikiuota teisingai, naudojant standartinę Go kalbos rikiavimo biblioteką
	if !sort.IsSorted(sort.IntSlice(a0)) || !sort.IsSorted(sort.Reverse(sort.IntSlice(a1))) {
		fmt.Println("sorting failed")
	}
}
