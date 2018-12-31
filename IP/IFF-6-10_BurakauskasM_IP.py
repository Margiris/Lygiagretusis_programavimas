from enum import Enum
from multiprocessing import Process, Queue

# dataFilename = "IFF-6-10_BurakauskasM_IP_dat_1.csv"
# dataFilename = "IFF-6-10_BurakauskasM_IP_dat_2.csv"
dataFilename = "IFF-6-10_BurakauskasM_IP_dat_3.csv"

resultsFilename = "IFF-6-10_BurakauskasM_IP_rez.txt"

headerLineCar = f"{'Nr.': <3} {'Gamintojas': <13} {'Modelis': <18} {'Metai': <5} {'Kaina': <9}\n"
headerLineOrder = f"{'Nr.': <3} {'Metai': <5} {'Kiekis': <6}\n"

separatorLineCar = "-" * 52 + "\n"
separatorLineOrder = "-" * 16 + "\n"

delimiter = ","

queueSize = 1


# Types of events.
class EventType(Enum):
    REMOVE = 0
    ADD = 1
    CONSUMER_FINISH = 10
    PRODUCER_FINISH = 11
    CONTROLLER_FINISH = 12


# Types of results.
class ResultType(Enum):
    FAIL = 0
    SUCCESS = 1


# Event class holds information about operation performed with an object, it's data and result of the action.
class Event:
    def __init__(this, subject, eventType, result=ResultType.FAIL):
        this.subject = subject
        this.type = eventType
        this.result = result

    # Changes result value
    def setResult(this, result):
        this.result = result

    # Returns formatted string with all event's data
    def ToString(this):
        if this.result is ResultType.SUCCESS:
            resultString = "Pavyko"
        elif this.result is ResultType.FAIL:
            resultString = "Nepavyko"
        else:
            resultString = ""

        if this.type is EventType.ADD:
            typeString = "pridėti   "
        elif this.type is EventType.REMOVE:
            typeString = "pašalinti"
        else:
            typeString = ""

        return f"{resultString: <8} {typeString: <9} {this.subject.ToString()}"


# Structure of Car type data
class Car:
    def __init__(this, manufacturer, model, year, price):
        this.manufacturer = manufacturer
        this.model = model
        this.year = year
        this.price = price

    # Returns formatted string with required properties of car object
    def ToString(this):
        return f"{this.manufacturer: <13} {this.model: <18} {this.year: <5} {this.price: >9.2f}\n"


# Structure of Order type data
class Order:
    def __init__(this, year, count):
        this.year = year
        this.count = count

    # Decreases count and returns true if object has count higher than 1, returns false otherwise
    def RemoveOne(this):
        if this.count > 1:
            this.count -= 1
            return True
        else:
            return False

    # Returns formatted string with required properties of order object
    def ToString(this):
        return f"{this.year: >5} {this.count: >6}\n"


# Main data structure
class Store:
    def __init__(this):
        this.availableCars = list()

    # Increases count of specified year in availableCars list if Order object with that year value already exists,
    # inserts order object to the availableCars list in a sorted manner otherwise
    def AddCar(this, newCar):
        index = 0

        for car in this.availableCars:
            if car.year == newCar.year:
                car.count += 1
                return True
            elif car.year > newCar.year:
                break
            index += 1

        this.availableCars.insert(index, Order(newCar.year, 1))

        return True

    # Removes order object from list and leaves no gap if RemoveOne method returns false
    def RemoveCar(this, order):
        for car in this.availableCars:
            if car.year == order.year:
                if not car.RemoveOne():
                    this.availableCars.remove(car)

                return True

        return False

    # Prints availableCars list to console
    def PrintAvailableCars(this):
        print("Laisvi automobiliai:")

        index = 0
        for car in this.availableCars:
            print(f"{index + 1: >2} {car.ToString()}", end='')
            index += 1

        print()

    # Appends availableCars list to file
    def WriteAvailableCars(this):
        with open(resultsFilename, "a") as resultsFile:
            resultsFile.write("Laisvi automobiliai:\n")

            if len(this.availableCars) > 0:
                resultsFile.write(headerLineOrder)
                resultsFile.write(separatorLineOrder)

                index = 0
                for car in this.availableCars:
                    resultsFile.write(f"{index + 1: >2} {car.ToString()}")
                    index += 1

                resultsFile.write(separatorLineOrder)
            else:
                resultsFile.write("-")


# Prints specified file to console in whole
def PrintFileToConsole(filename):
    with open(filename, "r") as resultsFile:
        print(resultsFile.read())


# Reads from file to Car and Order list of list type structures
def ReadData(filename):
    producersData = list()
    consumersData = list()

    with open(filename, "r") as dataFile:
        for groupLineCount in dataFile:
            carsData = list()
            ordersData = list()

            for lineIndex in range(0, int(groupLineCount)):
                currentValues = dataFile.readline().split(delimiter)
                if len(currentValues) == 4:
                    carsData.append(
                        Car(currentValues[0], currentValues[1], int(currentValues[2]), float(currentValues[3])))

                elif len(currentValues) == 2:
                    ordersData.append(Order(int(currentValues[0]), int(currentValues[1])))

            if len(carsData) > len(ordersData):
                producersData.append(carsData)
            else:
                consumersData.append(ordersData)

    return consumersData, producersData


# Writes Order and Car type data to file
def WriteData(consumers, producers):
    with open(resultsFilename, "w") as resultsFile:
        resultsFile.write(headerLineCar)
        resultsFile.write(separatorLineCar)

        for producer in producers:
            lineIndex = 1

            for car in producer:
                resultsFile.write(f"{lineIndex: >3} {car.ToString()}")
                lineIndex += 1

            resultsFile.writelines(separatorLineCar)

        resultsFile.write(headerLineOrder)
        resultsFile.write(separatorLineOrder)

        for consumer in consumers:
            lineIndex = 1

            for order in consumer:
                resultsFile.write(f"{lineIndex: >3} {order.ToString()}")
                lineIndex += 1

            resultsFile.write(separatorLineOrder)


# Consumer sends it's data to controller one by one until all data is sent.
def Consumer(orders, orderQueue):
    for order in orders:
        orderQueue.put(order)

    orderQueue.put(Event(None, EventType.CONSUMER_FINISH, ResultType.SUCCESS))


# Producer sends it's data to controller one by one until all data is sent.
def Producer(cars, carQueue):
    for car in cars:
        carQueue.put(car)

    carQueue.put(Event(None, EventType.PRODUCER_FINISH, ResultType.SUCCESS))


# Controller receives data from consumers or producers and puts it in corresponding unfinished data list.
# Then it iterates through those lists and tries to add and remove cars from main data structure. If successful,
# those items are removed from corresponding unfinished lists. All tries are logged by sending Event type object to
# Logger process.
# Controller finishes when finished consumers and producers counters reach total consumer and producer thread count.
def Controller(dataQueue, eventQueue, resultQueue, consumerThreadCount, producerThreadCount):
    store = Store()

    consumersFinished = 0
    producersFinished = 0

    unfinishedCars = list()
    unfinishedOrders = list()

    while consumersFinished < consumerThreadCount or producersFinished < producerThreadCount:

        message = dataQueue.get()

        if type(message) is Event:
            if message.type is EventType.CONSUMER_FINISH and message.result is ResultType.SUCCESS:
                consumersFinished += 1
            elif message.type is EventType.PRODUCER_FINISH and message.result is ResultType.SUCCESS:
                producersFinished += 1

        elif type(message) is Car:
            unfinishedCars.append(message)
        elif type(message) is Order:
            unfinishedOrders.append(message)

        for car in unfinishedCars:
            newEvent = Event(car, EventType.ADD)

            if store.AddCar(car):
                newEvent.setResult(ResultType.SUCCESS)
                unfinishedCars.remove(car)

            eventQueue.put(newEvent)

        for order in unfinishedOrders:
            newEvent = Event(Order(order.year, 1), EventType.REMOVE)

            if store.RemoveCar(order):
                newEvent.setResult(ResultType.SUCCESS)
                if not order.RemoveOne():
                    unfinishedOrders.remove(order)

            eventQueue.put(newEvent)

    eventQueue.put(Event(None, EventType.CONTROLLER_FINISH, ResultType.SUCCESS))
    resultQueue.put(store)


# Prints received log message to console. Finishes upon receiving Controller finish event.
def Logger(eventQueue):
    message = eventQueue.get()

    while message.type is not EventType.CONTROLLER_FINISH:
        print(message.ToString(), end='')
        message = eventQueue.get()


def main():
    # Read and write data to file
    consumers, producers = ReadData(dataFilename)
    WriteData(consumers, producers)

    # Create thread lists
    threadsConsumer = list()
    threadsProducer = list()
    threadsOther = list()

    # Create queues of size queueSize
    dataQueue = Queue(queueSize)
    eventQueue = Queue(queueSize)
    resultQueue = Queue(queueSize)

    # Construct and put consumer threads to list
    for consumer in consumers:
        threadsConsumer.append(Process(target=Consumer, args=(consumer, dataQueue,)))

    # Construct and put producer threads to list
    for producer in producers:
        threadsProducer.append(Process(target=Producer, args=(producer, dataQueue,)))

    # Construct and put controller and logger threads to list
    threadsOther.append(
        Process(target=Controller, args=(dataQueue, eventQueue, resultQueue, len(consumers), len(producers))))
    threadsOther.append(Process(target=Logger, args=(eventQueue,)))

    # Start all created threads
    for thread in threadsConsumer + threadsProducer + threadsOther:
        thread.start()

    # Get result and write it to file
    store = resultQueue.get()
    store.WriteAvailableCars()

    # Print results file to console
    PrintFileToConsole(resultsFilename)

    # Wait for all threads to finish
    # for thread in threadsConsumer + threadsProducer + threadsOther:
    #     thread.join()


if __name__ == "__main__":
    main()
