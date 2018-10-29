#include <string>
#include <vector>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

using namespace std;

const string data_filename = R"(IFF-6-10_BurakauskasM_L2a_dat.csv)";
const string results_filename = R"(IFF-6-10_BurakauskasM_L2a_rez.txt)";

const char delimiter = ',';
const int total_size = 100;

int data_count;

// int producer_thread_count;
// int consumer_thread_count;

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
};

class producer
{
    vector<car> cars_data_;

    public:
    explicit producer(const vector<car>& list_of_cars)
    {
        cars_data_ = list_of_cars;
    }

    void add_car(const car& new_car)
    {
        cars_data_.push_back(new_car);
    }

    void print_cars_data()
    {
        stringstream buffer;

        buffer << left << setw(4) << "Nr." << setw(15) << "Gamintojas" << setw(20) << "Modelis" << right << setw(5) <<
            "Metai" << setw(10) << fixed << setprecision(2) << "Kaina" << endl;
        buffer << string(54, '-') << "\n";

        auto line_index = 0;

        for (const auto& a_car : cars_data_)
        {
            buffer << left << setw(4) << ++line_index << setw(15) << a_car.manufacturer << setw(20) << a_car.model <<
                right << setw(5) << a_car.year << setw(10) << fixed << setprecision(2) << a_car.price << "\n";
        }

        buffer << string(54, '-') << "\n";

        ofstream results_file(results_filename, ios::app);
        results_file << buffer.str();
        cout << buffer.str();
    }

    // void produce()
    // {
    // }
};

class consumer
{
    vector<order> orders_data_;

    public:
    explicit consumer(const vector<order>& orders)
    {
        orders_data_ = orders;
    }

    void add_order(const order& new_order)
    {
        orders_data_.push_back(new_order);
    }

    void print_orders_data()
    {
        stringstream buffer;

        buffer << left << setw(4) << "Nr." << right << setw(6) << "Metai" << setw(7) << "Kiekis" << "\n";
        buffer << string(17, '-') << "\n";

        auto line_index = 0;

        for (const auto& an_order : orders_data_)
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

void read_producer_data(vector<producer>& producers, vector<consumer>& consumers)
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

void clear_results_file()
{
    ofstream file;
    file.open(results_filename, ofstream::out | ofstream::trunc);
    file.close();
}

int main()
{
    clear_results_file();

    vector<producer> producers;
    vector<consumer> consumers;

    read_producer_data(producers, consumers);

    for (auto& producer : producers)
    {
        producer.print_cars_data();
    }

    for (auto& consumer : consumers)
    {
        consumer.print_orders_data();
    }

    system("pause");
}
