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
const string delimiter = ";";

int producer_thread_count;
int consumer_thread_count;
const int total_size = 100;

struct car
{
    string manufacturer;
    string model;
    int year{};
    double price{};
};

struct item
{
    int year;
    int count;
};

vector<car> car_data;
vector<item> orders;
int data_count;

class monitor
{
    item available_cars_[total_size]{};
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
};

struct producer
{
};

struct consumer
{
};

void print_car(const car a_car)
{
    cout << left << setw(15) << a_car.manufacturer << setw(20) << a_car.model
        << right << setw(5) << a_car.year << setw(10) << fixed << setprecision(2) << a_car.price << "\n";
}

void read_data()
{
    ifstream file(data_filename);

    string current_line;

    while (getline(file, current_line))
    {
        vector<string> current_values;
        stringstream ss(current_line);
        string item_string;

        while (getline(ss, item_string, ','))
        {
            current_values.push_back(item_string);
        }

        if (current_values.size() == 4)
        {
            const auto manufacturer = current_values[0];
            const auto model = current_values[1];
            const int year = stol(current_values[2]);
            const double price = stold(current_values[3]);

            car_data.push_back(car{manufacturer, model, year, price});
        }
        else if (current_values.size() == 2)
        {
            const int year = stol(current_values[0]);
            const int count = stol(current_values[1]);

            orders.push_back(item{year, count});
        }
    }

    data_count = car_data.size();
}

int main()
{
    read_data();

    for (const auto& i : car_data)
    {
        print_car(i);
    }

    system("pause");
}
