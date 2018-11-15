#include <mpi.h>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdlib.h>

using namespace std;

const string data_filename = R"(.\..\L3data\IFF-6-10_BurakauskasM_L3_dat_1.csv)";
// const string data_filename = R"(.\..\L3data\IFF-6-10_BurakauskasM_L3_dat_2.csv)";
// const string data_filename = R"(.\..\L3data\IFF-6-10_BurakauskasM_L3_dat_3.csv)";
const string results_filename = R"(IFF-6-10_BurakauskasM_L3a_rez.txt)";

const char delimiter = ',';
// Size of the main data structure.
const int total_size = 8;
// Number of times consumer threads try to remove their items from available cars list.
const int allowed_tries_after_produce = 1;

const int producer_thread_count = 6;
const int consumer_thread_count = 5;

struct car
{
    string manufacturer;
    string model;
    int year{};
    double price{};
};

struct order
{
    int year;
    int count;
};

/**
 * \brief Class that contains shared data structure and methods for interaction with it.
 */
class monitor
{
    // Main data structure that producers write to and consumers remove from.
    order available_cars_[total_size]{};
    int available_cars_count_;

    // Arrays that indicate whether producer and consumer threads exist.
    bool producers_exist_[100]{};
    bool consumers_exit_[100]{};
    
    /**
     * \brief Finds index in available_cars_ array for specified year value in ascending order.
     * \param year Value to find index for.
     * \return Index of given parameter.
     */
    int calculate_index_for(const int year) const
    {
        if (available_cars_count_ == 0)
        {
            return 0;
        }

        for (auto i = 0; i < available_cars_count_; i++)
        {
            if (year <= available_cars_[i].year)
            {
                return i;
            }
        }

        return available_cars_count_;
    }

    /**
     * \brief Puts given year value in available_cars_ array:
     * if parameter value exists in array only increases count by 1,
     * else pushes other elements up an index and inserts year value to keep array sorted. Also sets count to 1.
     * \param year Value to insert in the array.
     */
    void list_car(const int year)
    {
        const auto index = calculate_index_for(year);

        if (year == available_cars_[index].year)
        {
            available_cars_[index].count++;
        }
        else
        {
            if (index != available_cars_count_)
            {
                for (auto i = available_cars_count_; i > index; i--)
                {
                    available_cars_[i] = available_cars_[i - 1];
                }
            }

            available_cars_[index].count = 1;
            available_cars_[index].year = year;
            available_cars_count_++;
        }
    }

    /**
     * \brief Removes specified year from the available_cars_ array:
     * if count is more than 1 only decreases count,
     * else removes specified parameter value and pushes other elements down an index to leave no gap.
     * In case that specified value was not found makes no changes and returns false.
     * \param year Value to remove from the array.
     * \return Whether the removal was made.
     */
    bool unlist_car(const int year)
    {
        for (auto i = 0; i < available_cars_count_; i++)
        {
            if (year == available_cars_[i].year)
            {
                if (available_cars_[i].count > 1)
                {
                    available_cars_[i].count--;
                }
                else
                {
                    for (auto o = i; o < available_cars_count_; o++)
                    {
                        available_cars_[o] = available_cars_[o + 1];
                    }

                    available_cars_count_--;
                }

                return true;
            }
        }

        return false;
    }

    public:
    /**
     * \brief Constructor
     */
    monitor()
    {
        available_cars_count_ = 0;

        for (auto& producer_exists : producers_exist_)
        {
            producer_exists = false;
        }
        for (auto& consumer_exists : consumers_exit_)
        {
            consumer_exists = false;
        }
    }

    /**
     * \brief Checks if any producers are producing.
     * \return boolean result.
     */
    bool is_producing()
    {
        for (auto& producer_exists : producers_exist_)
        {
            if (producer_exists)
            {
                return true;
            }
        }

        return false;
    }

    /**
     * \brief  Checks if any consumers are consuming.
     * \return boolean result.
     */
    bool is_consuming()
    {
        for (auto& consumer_exists : consumers_exit_)
        {
            if (consumer_exists)
            {
                return true;
            }
        }

        return false;
    }

    /**
     * \brief Sets that specified producer is producing.
     * \param index Producer number
     */
    void announce_producer(const int index)
    {
        producers_exist_[index] = true;
    }

    /**
     * \brief Sets that specified producer is no longer producing.
     * \param index Producer number
     */
    void suppress_producer(const int index)
    {
        producers_exist_[index] = false;
    }

    /**
     * \brief Sets that specified consumer is consuming.
     * \param index Consumer number
     */
    void announce_consumer(const int index)
    {
        producers_exist_[index] = true;
    }

    /**
     * \brief Sets that specified consumer is no longer consuming.
     * \param index Consumer number
     */
    void suppress_consumer(const int index)
    {
        producers_exist_[index] = false;
    }

    /**
     * \brief If any consumers exist waits for available_cars_ array to be not full and then if
     * array is not full calls function list_car then notifies other threads upon completion.
     * \param new_car Car object to add
     */
    void add_available_car(const car new_car)
    {

        if (is_consuming() && available_cars_count_ >= total_size)
        {
        }

        if (available_cars_count_ < total_size)
        {
            list_car(new_car.year);

        }
        else
        {
        }
    }

    /**
     * \brief If any producers exist waits for available_cars_ array to be not empty and then if
     * array is not empty calls function unlist_car then notifies other threads upon completion.
     * \param order1 Order object to remove.
     * \return Whether the specified order was removed.
     */
    bool remove_available_car(const order order1)
    {
        auto was_removed = false;

        if (is_producing() && available_cars_count_ <= 0)
        {
        }

        if (available_cars_count_ > 0)
        {
            was_removed = unlist_car(order1.year);
        }

        return was_removed;
    }

    /**
     * \brief Prints available_cars_ array to buffer which is then appended to results file and console.
     */
    void print_available_cars()
    {
        stringstream buffer;

        buffer << "Laisvi automobiliai:\n";

        if (available_cars_count_ > 0)
        {
            buffer << left << setw(4) << "Nr." << right << setw(6) << "Metai" << setw(7) << "Kiekis" << "\n";
            buffer << string(17, '-') << "\n";

            for (auto i = 0; i < total_size; i++)
            {
                if (available_cars_[i].count > 0)
                {
                    buffer << left << setw(4) << i + 1 << right << setw(6) << available_cars_[i].year <<
                        setw(7) << available_cars_[i].count << "\n";
                }
                else break;
            }

            buffer << string(17, '-') << "\n";
        }
        else { buffer << "-\n"; }

        ofstream results_file(results_filename, ios::app);
        results_file << buffer.str();
        cout << buffer.str();

        results_file.close();
    }
};

monitor global_monitor;

class producer
{
    public:

    vector<car> cars_data;

    explicit producer(const vector<car>& list_of_cars)
    {
        cars_data = list_of_cars;
    }

    /**
     * \brief Prints cars_data array to buffer which is then appended to results file and console.
     */
    void print_cars_data()
    {
        stringstream buffer;

        buffer << left << setw(4) << "Nr." << setw(15) << "Gamintojas" << setw(20) << "Modelis" << right << setw(5) <<
            "Metai" << setw(10) << fixed << setprecision(2) << "Kaina" << endl;
        buffer << string(54, '-') << "\n";

        auto line_index = 0;

        for (const auto& a_car : cars_data)
        {
            buffer << left << setw(4) << ++line_index << setw(15) << a_car.manufacturer << setw(20) << a_car.model <<
                right << setw(5) << a_car.year << setw(10) << fixed << setprecision(2) << a_car.price << "\n";
        }

        buffer << string(54, '-') << "\n";

        ofstream results_file(results_filename, ios::app);
        results_file << buffer.str();
        cout << buffer.str();
    }
};

class consumer
{
    public:

    vector<order> orders_data;

    explicit consumer(const vector<order>& orders)
    {
        orders_data = orders;
    }

    /**
     * \brief Prints orders_data array to buffer which is then appended to results file and console.
     */
    void print_orders_data()
    {
        stringstream buffer;

        buffer << left << setw(4) << "Nr." << right << setw(6) << "Metai" << setw(7) << "Kiekis" << "\n";
        buffer << string(17, '-') << "\n";

        auto line_index = 0;

        for (const auto& an_order : orders_data)
        {
            buffer << left << setw(4) << ++line_index << right << setw(6) << an_order.year << setw(7) << an_order.count
                << "\n";
        }

        buffer << string(17, '-') << "\n";

        ofstream results_file(results_filename, ios::app);
        results_file << buffer.str();
        cout << buffer.str();
    }
};

/**
 * \brief Reads data from data file to producers and consumers object lists.
 * \param producers List of producer objects.
 * \param consumers List of consumer objects.
 */
void read_data(vector<producer>& producers, vector<consumer>& consumers)
{
    ifstream data_file(data_filename);

    string current_line;

    while (getline(data_file, current_line))
    {
        const int elements_count = stol(current_line);

        vector<car> cars_data;
        vector<order> orders_data;

        for (auto i = 0; i < elements_count; i++)
        {
            getline(data_file, current_line);

            vector<string> current_values;
            stringstream ss(current_line);
            string car_property_value;

            while (getline(ss, car_property_value, delimiter))
            {
                current_values.push_back(car_property_value);
            }

            if (current_values.size() == 4)
            {
                const auto manufacturer = current_values[0];
                const auto model = current_values[1];
                const int year = stol(current_values[2]);
                const double price = stold(current_values[3]);

                cars_data.push_back(car{manufacturer, model, year, price});
            }
            else if (current_values.size() == 2)
            {
                const int year = stol(current_values[0]);
                const int count = stol(current_values[1]);

                orders_data.push_back(order{year, count});
            }
        }

        (cars_data.size() > orders_data.size()
             ? producers.emplace_back(cars_data)
             : consumers.emplace_back(orders_data));
    }
}

/**
 * \brief Calls print method of each producer and consumer in provided lists.
 * \param producers List of producer objects.
 * \param consumers List of consumer objects.
 */
void write_data(vector<producer>& producers, vector<consumer>& consumers)
{
    for (auto& producer : producers)
    {
        producer.print_cars_data();
    }

    for (auto& consumer : consumers)
    {
        consumer.print_orders_data();
    }
}

/**
 * \brief Calls monitor function to add each car from producer.cars_data object.
 * \param producer1 Producer object.
 */
void produce(int rank)
{
    cout << rank << " trying to produce\n";
    vector<car> cars;
    producer producer1(cars);

    for (const auto& a_car : producer1.cars_data)
    {
        global_monitor.add_available_car(a_car);
    }
}

/**
 * \brief While producers are still producing and specified number of times after they are no
 * longer producing calls monitor function to remove orders in consumer.orders_data object.
 * \param consumer1 Consumer object.
 */
void consume(int rank)
{
    cout << rank << " trying to consume\n";
    vector<order> orders;
    consumer consumer1(orders);

    auto tries_after_produce = 0;

    while ((global_monitor.is_producing() || tries_after_produce < allowed_tries_after_produce)
           && !consumer1.orders_data.empty())
    {
        auto was_removed = false;

        for (auto i = 0; i < int(consumer1.orders_data.size()); i++)
        {
            const auto was_removed_once = global_monitor.remove_available_car(consumer1.orders_data[i]);

            if (was_removed_once)
            {
                was_removed = true;

                if (consumer1.orders_data[i].count > 1)
                {
                    consumer1.orders_data[i].count--;
                }
                else
                {
                    consumer1.orders_data.erase(consumer1.orders_data.begin() + i);
                }
            }
        }

        if (!global_monitor.is_producing() && !was_removed)
        {
            tries_after_produce++;
        }
    }
}

/**
 * \brief Clears results file.
 */
void clear_results_file()
{
    ofstream file;
    file.open(results_filename, ofstream::out | ofstream::trunc);
    file.close();
}

// int main()
// {
//     vector<producer> producers;
//     vector<consumer> consumers;
//
//     read_data(producers, consumers);
//
//     // If data structures are empty, prints an error message and exits.
//     if (producers.empty())
//     {
//         cout << "No data, exiting. ";
//         system("pause");
//         return -1;
//     }
//
//     write_data(producers, consumers);
//
//     start_threads(producers, consumers);
//
//     global_monitor.print_available_cars();
//
//     system("pause");
// }

int main(int argc, char* argv[])
{
    clear_results_file();

    MPI_Init(&argc, &argv);

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Controller thread
    if (rank == 0)
    {
        char hello_str[] = "Hello World";
        MPI_Send(hello_str, 12, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    }
    // Data manager thread
    else if (rank == 1)
    {
        cout << "\n";

        char hello_str[12];
        MPI_Recv(hello_str, 12, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cout << "Rank 1 received string " << hello_str << " from Rank 0\n";

        cout << "\n";
    }
    // Producer threads
    else if (rank < 2 + producer_thread_count)
    {
        produce(rank);
    }
    // Consumer threads
    else if (rank < 2 + producer_thread_count + consumer_thread_count)
    {
        consume(rank);
    }

    MPI_Finalize();
    return 0;
}
