#include <string>
#include <vector>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <omp.h>

using namespace std;

// const string data_filename = R"(.\..\L2data\IFF-6-10_BurakauskasM_L2_dat_1.csv)";
// const string data_filename = R"(.\..\L2data\IFF-6-10_BurakauskasM_L2_dat_2.csv)";
const string data_filename = R"(.\..\L2data\IFF-6-10_BurakauskasM_L2_dat_3.csv)";
const string results_filename = R"(IFF-6-10_BurakauskasM_L2b_rez.txt)";

const char delimiter = ',';
const int total_size = 100;
const int allowed_tries_after_produce = 0;

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

class monitor
{
    order available_cars_[total_size]{};
    int available_cars_count_;
    bool producers_exist_[total_size]{};

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
    monitor()
    {
        available_cars_count_ = 0;

        for (auto& producer_exists : producers_exist_)
        {
            producer_exists = false;
        }
    }

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

    void announce_producer(const int index)
    {
        producers_exist_[index] = true;
    }

    void suppress_producer(const int index)
    {
        producers_exist_[index] = false;
    }

    void add_available_car(const car new_car)
    {
        // cv_.wait(guard, [=] { return available_cars_count_ < total_size; });
        #pragma omp critical (order_array_critical)
        {
            // list_car(new_car.year);
            const auto index = calculate_index_for(new_car.year);

            if (new_car.year == available_cars_[index].year)
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
                available_cars_[index].year = new_car.year;
                available_cars_count_++;
            }
        }

        // cv_.notify_all();
    }

    bool remove_available_car(const order order1)
    {
        // if (is_producing())
        // {
        //     cv_.wait(guard, [=] { return available_cars_count_ > 0; });
        // }
        #pragma omp critical (order_array_critical)
        {
            // const auto was_removed = unlist_car(order1.year);
            //
            // return was_removed;
            for (auto i = 0; i < available_cars_count_; i++)
            {
                if (order1.year == available_cars_[i].year)
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

        // cv_.notify_all();
    }

    void print_available_cars() const
    {
        stringstream buffer;

        buffer << "Laisvi automobiliai:\n";

        if (available_cars_count_ > 0)
        {
            buffer << left << setw(4) << "Nr." << right << setw(6) << "Metai" << setw(7) << "Kiekis" << "\n";
            buffer << string(17, '-') << "\n";

            auto line_index = 0;

            while (available_cars_[line_index++].year > 0)
            {
                buffer << left << setw(4) << line_index << right << setw(6) << available_cars_[line_index - 1].year <<
                    setw(7) << available_cars_[line_index - 1].count << "\n";
            }

            buffer << string(17, '-') << "\n";
        }
        else { buffer << "-\n"; }

        ofstream results_file(results_filename, ios::app);
        results_file << buffer.str();
        cout << buffer.str();
    }
};

monitor global_monitor;

class producer
{
    public:

    vector<car> cars_data;

    explicit producer(vector<car>& list_of_cars)
    {
        cars_data = list_of_cars;
    }

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

    explicit consumer(vector<order>& orders)
    {
        orders_data = orders;
    }

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

void produce(producer producer1)
{
    for (const auto& a_car : producer1.cars_data)
    {
        global_monitor.add_available_car(a_car);
    }
}

void consume(consumer consumer1)
{
    cout << "trying to remove\n";
    auto tries_after_produce = 0;

    while ((global_monitor.is_producing() || tries_after_produce < allowed_tries_after_produce)
           && !consumer1.orders_data.empty())
    {
        for (auto i = 0; i < int(consumer1.orders_data.size()); i++)
        {
            const auto was_removed = global_monitor.remove_available_car(consumer1.orders_data[i]);

            if (was_removed)
            {
                if (consumer1.orders_data[i].count > 1)
                {
                    consumer1.orders_data[i].count--;
                }
                else
                {
                    consumer1.orders_data.erase(consumer1.orders_data.begin() + i);
                }
            }
            else if (!global_monitor.is_producing())
            {
                tries_after_produce++;
            }
        }
    }
}

void clear_results_file()
{
    ofstream file;
    file.open(results_filename, ofstream::out | ofstream::trunc);
    file.close();
}

void start_threads(const vector<producer>& producers, const vector<consumer>& consumers)
{
    const int thread_count = producers.size() + consumers.size();

    #pragma omp parallel num_threads(thread_count)
    {
        const auto thread_id = omp_get_thread_num();

        if (thread_id < int(producers.size()))
        {
            cout << "trying to add, id " << thread_id << "\n";
            global_monitor.announce_producer(thread_id);
            produce(producers[thread_id]);
            global_monitor.suppress_producer(thread_id);
        }
        else
        {
            consume(consumers[thread_id - producers.size()]);
        }
    }
}

void wait_for_threads(thread producer_threads[], const vector<producer>& producers,
                      thread consumer_threads[], const vector<consumer>& consumers)
{
    for (auto i = 0; i < int(producers.size()); i++)
    {
        producer_threads[i].join();

        global_monitor.suppress_producer(i);
    }

    for (auto i = 0; i < int(consumers.size()); i++)
    {
        consumer_threads[i].join();
    }
}

int main()
{
    clear_results_file();

    vector<producer> producers;
    vector<consumer> consumers;

    read_data(producers, consumers);

    if (producers.empty())
    {
        cout << "No data, exiting. ";
        system("pause");
        return -1;
    }

    write_data(producers, consumers);

    // const auto producer_threads = new thread[producers.size()];
    // const auto consumer_threads = new thread[producers.size()];

    start_threads(producers, consumers);
    //wait_for_threads(producer_threads, producers, consumer_threads, consumers);

    global_monitor.print_available_cars();

    // delete [] producer_threads;
    // delete [] consumer_threads;

    system("pause");

    return 0;
}
