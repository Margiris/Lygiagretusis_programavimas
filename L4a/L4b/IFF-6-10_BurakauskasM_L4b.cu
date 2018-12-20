
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "iostream"
#include "string"
#include "fstream"
#include <thread>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;
using namespace thrust;

const char data_filename[] = "../../L4data/IFF-6-10_BurakauskasM_L4.csv";
const char results_filename[] = "../../L4data/IFF-6-10_BurakauskasM_L4b_rez.txt";

const int max_string_length = 47;
const int group_count = 5;
const int n = 30;

// Structure of Car type data
struct car
{
	char manufacturer[max_string_length];
	char model[max_string_length];
	int year;
	double price;

	// Default constructor
	// Called from either CPU or GPU
	__host__ __device__ car() {};

	// Constructor that assigns passed values to properties
	// Called from either CPU or GPU
	__host__ __device__ car(char manufacturerNew[], char modelNew[], int yearNew, double priceNew) {
		strcpy(manufacturer, manufacturerNew);
		strcpy(model, modelNew);
		year = yearNew;
		price = priceNew;
	}
};

// Main data array
car Cars_data[group_count][n];
// Main results array
car Car_results[n];
// Main data array's elements' length
int Car_data_sizes[group_count];

// Clears results file
// Run on CPU
void clear_results_file()
{
	ofstream file;
	file.open(results_filename, ofstream::out | ofstream::trunc);
	file.close();
}

// Reads from file to Car structure
// Run on CPU
void read_data()
{
	ifstream data_file(data_filename);

	for (auto i = 0; i < group_count; i++)
	{
		data_file >> Car_data_sizes[i];

		for (auto j = 0; j < Car_data_sizes[i]; j++)
		{
			char manufacturer[100], model[100];

			data_file >> manufacturer >> model >> Cars_data[i][j].year >> Cars_data[i][j].price;
			strcpy_s(Cars_data[i][j].manufacturer, manufacturer);
			strcpy_s(Cars_data[i][j].model, model);
		}
	}

	data_file.close();
}

// Writes Car type data to file
// Run on CPU
void write_data() {
	stringstream buffer;

	buffer << left << setw(4) << "Nr." << setw(15) << "Gamintojas" << setw(20) << "Modelis" << right << setw(5) <<
		"Metai" << setw(10) << fixed << setprecision(2) << "Kaina" << endl;
	buffer << string(54, '-') << "\n";

	for (auto i = 0; i < group_count; i++)
	{
		auto line_index = 0;

		for (auto j = 0; j < Car_data_sizes[i]; j++)
		{
			buffer << setw(3) << ++line_index << " " << left << setw(15) << Cars_data[i][j].manufacturer << setw(20) << Cars_data[i][j].model <<
				right << setw(5) << Cars_data[i][j].year << setw(10) << fixed << setprecision(2) << Cars_data[i][j].price << "\n";
		}

		buffer << string(54, '-') << "\n";
	}

	ofstream results_file(results_filename, ios::app);
	results_file << buffer.str();
	cout << buffer.str();
}

// Appends results data of type Car to file
// Run on CPU
void write_result() {
	stringstream buffer;

	buffer << left << setw(4) << "Nr." << setw(max_string_length) << "Gamintojas" << setw(max_string_length) << "Modelis" << right << setw(7) <<
		"Metai" << setw(12) << fixed << setprecision(2) << "Kaina" << endl;
	buffer << string(23 + max_string_length * 2, '-') << "\n";

	auto line_index = 0;

	for (auto i = 0; i < n; i++)
	{
		if (Car_results[i].year != 0)
		{
			buffer << setw(3) << ++line_index << " " << left << setw(max_string_length) << Car_results[i].manufacturer << setw(max_string_length) << Car_results[i].model <<
				right << setw(7) << Car_results[i].year << setw(12) << fixed << setprecision(2) << Car_results[i].price << "\n";
		}
	}

	buffer << string(23 + max_string_length * 2, '-') << "\n";

	ofstream results_file(results_filename, ios::app);
	results_file << buffer.str();
	cout << buffer.str();
}

// strcpy function replacement that is called from GPU and run on GPU
__device__ char * custom_strcpy(char *destination, const char *source) {
	auto i = 0;

	do {
		destination[i] = source[i];
	} while (source[i++] != 0);

	return destination;
}

// strcat function replacement that is called from GPU and run on GPU
__device__ char * custom_strcat(char *destination, const char *source) {
	auto i = 0;

	while (destination[i] != 0)
		i++;

	custom_strcpy(destination + i, source);

	return destination;
}

struct functor {
	// Add Car type objects' elements to accumulator object by concatenating strings and summing numeric values
	// Called and run on GPU
	__device__ car operator()(car accumulator, car item) {
		const auto manufacturer = accumulator.manufacturer;
		custom_strcat(manufacturer, item.manufacturer);

		const auto model = accumulator.model;
		custom_strcat(model, item.model);

		accumulator.year += item.year;
		accumulator.price += item.price;

		return accumulator;
	}
};

int main()
{
	clear_results_file();

	// Read data from and write it to file
	read_data();
	write_data();

	// Create vector on CPU memory to store data
	host_vector<car> host_car[n];
	// Create vector on GPU memory to store data
	device_vector<car> device_car[n];

	// Add data from main data array to host vector and copy it to GPU memory by assigning it's value to device vector
	for (auto i = 0; i < n; i++)
	{
		for (auto o = 0; o < group_count; o++)
		{
			host_car[i].push_back(Cars_data[o][i]);
		}

		device_car[i] = host_car[i];
	}

	// Call reduce function for each vector in device vector array.
	// Reduce function uses operatot() override in functor struct
	for (auto i = 0; i < n; i++)
	{
		Car_results[i] = reduce(device_car[i].begin(), device_car[i].end(), car("", "", 0, 0), functor());
	}

	// Write results to file
	write_result();

	// Pause console window
	std::system("pause");
	return 0;
}
