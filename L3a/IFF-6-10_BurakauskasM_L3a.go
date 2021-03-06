package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

const dataFilename = "../L3data/IFF-6-10_BurakauskasM_L3_dat_1.csv"
// const dataFilename = "../L3data/IFF-6-10_BurakauskasM_L3_dat_2.csv"
// const dataFilename = "../L3data/IFF-6-10_BurakauskasM_L3_dat_3.csv"
// const dataFilename = "../L3data/IFF-6-10_BurakauskasM_L3_dat_4.csv"

const resultsFilename = "IFF-6-10_BurakauskasM_L3a_rez.txt"

var headerLineCar = fmt.Sprintf("%-3s %-14s %-19s %-6s %-8s", "Nr.", "Gamintojas", "Modelis", "Metai", "Kaina")
var headerLineOrder = fmt.Sprintf("%-3s %-6s %-6s", "Nr.", "Metai", "Kiekis")

var separatorLineCar = strings.Repeat("-", 54)
var separatorLineOrder = strings.Repeat("-", 17)

const delimiter = ","

// Size of the main data structure.
const totalSize = 30

// Number of times consumer threads try to remove their items from available cars list.
const allowedTriesAfterProduce = 1

// Structure of Car type data
type Car struct {
	manufacturer string
	model        string
	year         int
	price        float64
	threadIndex  int
}

// Returns formatted string with required properties of car object
func Car2str(car Car) string {
	return fmt.Sprintf("%-14s %-19s %4d %10.2f\n", car.manufacturer, car.model, car.year, car.price)
}

// Structure of Order type data
type Order struct {
	year        int
	count       int
	threadIndex int
}

// Returns formatted string with required properties of order object
func Order2str(order Order) string {
	return fmt.Sprintf("%5d %7d\n", order.year, order.count)
}

// Checks if error was null and panics if not
func check(e error) {
	if e != nil {
		panic(e)
	}
}

// Reads from file to Car and Order types
func ReadData() ([][]Order, [][]Car) {
	dataFilePath, _ := filepath.Abs(dataFilename)
	
	dataFile, err := os.Open(dataFilePath)
	check(err)
	defer dataFile.Close()
	
	scanner := bufio.NewScanner(dataFile)
	
	var producerData [][]Car
	var consumerData [][]Order
	
	for scanner.Scan() {
		var carsData []Car
		var ordersData []Order
		
		var elementsCount, _ = strconv.Atoi(scanner.Text())
		
		for i := 0; i < elementsCount; i++ {
			scanner.Scan()
			
			var currentValues = strings.Split(scanner.Text(), delimiter)
			
			if len(currentValues) == 4 {
				var m = currentValues[0]
				var o = currentValues[1]
				var y, _ = strconv.Atoi(currentValues[2])
				var p, _ = strconv.ParseFloat(currentValues[3], 64)
				
				carsData = append(carsData, Car{m, o, y, p, -1})
				
			} else if len(currentValues) == 2 {
				var y, _ = strconv.Atoi(currentValues[0])
				var c, _ = strconv.Atoi(currentValues[1])
				
				ordersData = append(ordersData, Order{y, c, -1})
			}
		}
		
		if len(carsData) > len(ordersData) {
			producerData = append(producerData, carsData)
		} else {
			consumerData = append(consumerData, ordersData)
		}
	}
	
	return consumerData, producerData
}

// Writes Order and Car type data to file
func WriteData(consumers [][]Order, producers [][]Car) {
	var buffer strings.Builder
	
	buffer.WriteString(headerLineCar + "\n")
	buffer.WriteString(separatorLineCar + "\n")
	
	for _, producer := range producers {
		var lineIndex = 1
		
		for _, car := range producer {
			buffer.WriteString(fmt.Sprintf("%2d  "+Car2str(car), lineIndex))
			lineIndex++
		}
		
		buffer.WriteString(separatorLineCar + "\n")
	}
	
	buffer.WriteString(headerLineOrder + "\n")
	buffer.WriteString(separatorLineOrder + "\n")
	
	for _, consumer := range consumers {
		var lineIndex = 1
		
		for _, order := range consumer {
			buffer.WriteString(fmt.Sprintf("%2d  "+Order2str(order), lineIndex))
			lineIndex++
		}
		
		buffer.WriteString(separatorLineOrder + "\n")
	}
	
	fmt.Println(buffer.String())
	
	resultsFile, err := os.Create(resultsFilename)
	check(err)
	defer resultsFile.Close()
	_, err = resultsFile.WriteString(buffer.String())
	check(err)
}

// Appends Order data to file
func WriteResults(results []Order) {
	var buffer strings.Builder
	
	//noinspection SpellCheckingInspection
	buffer.WriteString("Laisvi automobiliai:\n")
	
	if results[0].year != 0 {
		buffer.WriteString(headerLineOrder + "\n")
		buffer.WriteString(separatorLineOrder + "\n")
		
		for i := 0; results[i].year != 0; i++ {
			buffer.WriteString(fmt.Sprintf("%2d  "+Order2str(results[i]), i+1))
		}
	}
	
	buffer.WriteString(separatorLineOrder + "\n")
	
	fmt.Println(buffer.String())
	
	resultsFile, err := os.OpenFile(resultsFilename, os.O_APPEND|os.O_WRONLY, 0600)
	check(err)
	defer resultsFile.Close()
	
	_, err = resultsFile.WriteString(buffer.String())
	check(err)
}

// Returns int type number representing an index in Order type array
func FindIndex(availableCars []Order, availableCarsCount int, year int) int {
	if availableCarsCount == 0 {
		return 0
	}
	
	for i := 0; i < availableCarsCount; i++ {
		if year <= availableCars[i].year {
			return i
		}
	}
	
	return availableCarsCount
}

// Increases index of specified year in Order array if Order object with that year value already exists, inserts new Order type object in a sorted manner otherwise
func AddCar(availableCars []Order, availableCarsCount int, year int) ([]Order, int, bool) {
	var index = FindIndex(availableCars, availableCarsCount, year)
	
	if year == availableCars[index].year {
		availableCars[index].count++
	} else {
		if index != availableCarsCount {
			
			for i := availableCarsCount; i > index; i-- {
				availableCars[i] = availableCars[i-1]
			}
		}
		
		availableCars[index].count = 1
		availableCars[index].year = year
		availableCarsCount++
	}
	
	return availableCars, availableCarsCount, true
}

// Decreases index of specified year in Order array if Order object with that year value has count higher than 1, removes new Order type object and leaves no gap otherwise
func RemoveCar(availableCars []Order, availableCarsCount int, year int) ([]Order, int, bool) {
	for i := 0; i < availableCarsCount; i++ {
		if year == availableCars[i].year {
			if availableCars[i].count > 1 {
				availableCars[i].count--
			} else {
				for o := i; o < availableCarsCount; o++ {
					availableCars[o] = availableCars[o+1]
				}
				
				availableCarsCount--
			}
			return availableCars, availableCarsCount, true
		}
	}
	
	return availableCars, availableCarsCount, false
}

// Synchronizer object to wait for threads' finish
var synchronizer = sync.WaitGroup{}

// Data manager gets data by calling ReadData, sends count of consumers and producers to main, then receives channels for consumers and producers.
// Later sends data using those channels.
// Finally receives results  of Order type and calls WriteResults to output them.
func DataManager(countOutputChan chan<- int, orderOutputChansChan chan []chan []Order, carOutputChansChan chan []chan []Car, resultInputChan <-chan []Order) {
	defer synchronizer.Done()
	
	var consumersData, producersData = ReadData()
	WriteData(consumersData, producersData)
	
	countOutputChan <- len(consumersData)
	countOutputChan <- len(producersData)
	
	var orderOutputChans = <-orderOutputChansChan
	var carOutputChans = <-carOutputChansChan
	
	for i := 0; i < len(consumersData); i++ {
		orderOutputChans[i] <- consumersData[i]
	}
	for i := 0; i < len(producersData); i++ {
		carOutputChans[i] <- producersData[i]
	}
	
	var results = <-resultInputChan
	WriteResults(results)
}

// Consumer receives data from data manager, and sends it's data to controller one by one until all data is sent or all producers finish.
func Consumer(dataInputChan <-chan []Order, dataOutputChan chan<- Order, responseChan <-chan bool, threadIndex int) {
	defer synchronizer.Done()
	
	var orders = <-dataInputChan
	
	var triesAfterProduce = 0
	var producersExist = false
	
	for triesAfterProduce < allowedTriesAfterProduce {
		var wasRemoved = false
		
		for i := 0; i < len(orders); i++ {
			orders[i].threadIndex = threadIndex
			dataOutputChan <- orders[i]
			
			var wasRemovedOnce = <-responseChan
			producersExist = <-responseChan
			
			if wasRemovedOnce {
				wasRemoved = true
				
				if orders[i].count > 1 {
					orders[i].count--
				} else {
					orders = append(orders[:i], orders[i+1:]...)
				}
			}
		}
		
		if (!producersExist && !wasRemoved) || len(orders) <= 0 {
			triesAfterProduce++
		}
	}
	
	dataOutputChan <- Order{-1, -1, threadIndex}
}

// Producer receives data from data manager, and sends it's data to controller one by one until all data is sent.
func Producer(dataInputChan <-chan []Car, dataOutputChan chan<- Car, responseChan <-chan bool, threadIndex int) {
	defer synchronizer.Done()
	
	var cars = <-dataInputChan
	cars = append(cars, Car{"", "", -1, -1, threadIndex})
	
	for i := 0; i < len(cars); i++ {
		cars[i].threadIndex = threadIndex
		dataOutputChan <- cars[i]
		
		var wasAdded = <-responseChan
		if !wasAdded {
			i--
		}
	}
	
}

// Controller receives data from consumers or producers and removes or adds that data to main data structure respectively until there are no more consumers and producers.
// Then sends a response back to thread that the data was received from.
// Finally sends main data structure to data manager.
func Controller(consumersCount int, producersCount int, orderInputChan <-chan Order, carInputChan chan Car, responseOutputChanToConsumers []chan bool, responseOutputChanToProducers []chan bool, resultOutputChan chan<- []Order) {
	defer synchronizer.Done()
	
	var availableCars = make([]Order, totalSize)
	var availableCarsCount = 0
	
	for consumersCount > 0 || producersCount > 0 {
		select {
		case message := <-carInputChan:
			var senderIndex = message.threadIndex
			
			if message.year == -1 {
				producersCount--
				responseOutputChanToProducers[senderIndex] <- true
				fmt.Println("producer " + strconv.Itoa(message.threadIndex) + " finished, " + strconv.Itoa(producersCount) + " left.")
			} else {
				var wasAdded bool
				availableCars, availableCarsCount, wasAdded = AddCar(availableCars, availableCarsCount, message.year)
				responseOutputChanToProducers[senderIndex] <- wasAdded
			}
		
		case message := <-orderInputChan:
			var senderIndex = message.threadIndex
			
			if message.year == -1 {
				consumersCount--
				fmt.Println("consumer " + strconv.Itoa(message.threadIndex) + " finished, " + strconv.Itoa(consumersCount) + " consumers left.")
			} else {
				var wasRemoved bool
				availableCars, availableCarsCount, wasRemoved = RemoveCar(availableCars, availableCarsCount, message.year)
				responseOutputChanToConsumers[senderIndex] <- wasRemoved
				responseOutputChanToConsumers[senderIndex] <- producersCount > 0
			}
		}
	}
	
	resultOutputChan <- availableCars
}

func main() {
	// Create channels
	var mainToDataManagerForConsumers = make(chan []chan []Order)
	var mainToDataManagerForProducers = make(chan []chan []Car)
	var dataManagerToMain = make(chan int)
	var dataManagerToConsumer []chan []Order
	var dataManagerToProducer []chan []Car
	var consumersToController = make(chan Order)
	var producersToController = make(chan Car)
	var controllerToConsumer []chan bool
	var controllerToProducer []chan bool
	var controllerToDataManager = make(chan []Order)
	
	// Start data manager thread to read data
	synchronizer.Add(1)
	go DataManager(dataManagerToMain, mainToDataManagerForConsumers, mainToDataManagerForProducers, controllerToDataManager)
	
	// Receive how many consumers and producers there are
	var consumersCount = <-dataManagerToMain
	var producersCount = <-dataManagerToMain
	
	// Create channels for each consumer thread and start them.
	for i := 0; i < consumersCount; i++ {
		dataManagerToConsumer = append(dataManagerToConsumer, make(chan []Order))
		controllerToConsumer = append(controllerToConsumer, make(chan bool))
		synchronizer.Add(1)
		go Consumer(dataManagerToConsumer[i], consumersToController, controllerToConsumer[i], i)
	}
	
	// Send channels to data manager that it will use to send data to consumers
	mainToDataManagerForConsumers <- dataManagerToConsumer
	
	// Create channels for each producer thread and start them.
	for i := 0; i < producersCount; i++ {
		dataManagerToProducer = append(dataManagerToProducer, make(chan []Car))
		controllerToProducer = append(controllerToProducer, make(chan bool))
		synchronizer.Add(1)
		go Producer(dataManagerToProducer[i], producersToController, controllerToProducer[i], i)
	}
	
	// Send channels to data manager that it will use to send data to consumers
	mainToDataManagerForProducers <- dataManagerToProducer
	
	synchronizer.Add(1)

	// Start controller thread
	go Controller(consumersCount, producersCount, consumersToController, producersToController, controllerToConsumer, controllerToProducer, controllerToDataManager)
	
	// Wait for all threads to finish
	synchronizer.Wait()
}
