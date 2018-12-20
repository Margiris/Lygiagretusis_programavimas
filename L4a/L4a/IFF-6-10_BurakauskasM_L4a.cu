
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "iostream"
#include "string"
#include "fstream"
#include <thread>
#include <sstream>
#include <iomanip>
#include <mutex>

using namespace std;

const char data_filename[] = "../../L4data/IFF-6-10_BurakauskasM_L4.csv";
const char results_filename[] = "../../L4data/IFF-6-10_BurakauskasM_L4a_rez.txt";

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

// Add Car type objects' elements to result object by concatenating strings and summing numeric values
// Run on GPU, called from CPU
__global__ void add(car** Pa, car* result) {
	const int idx = threadIdx.x;
	result[idx].price = 0;

	for (auto i = 0; i < group_count; i++)
	{
		const auto manufacturer = result[idx].manufacturer;
		custom_strcat(manufacturer, Pa[i][idx].manufacturer);

		const auto model = result[idx].model;
		custom_strcat(model, Pa[i][idx].model);

		result[idx].year += Pa[i][idx].year;
		result[idx].price += Pa[i][idx].price;
	}
}

int main()
{
	clear_results_file();

	// Read data from and write it to file
	read_data();
	write_data();

	int data_array_size = n * sizeof(car);

	// Create pointer of type Car array on GPU memory
	car **CUDA_car;
	// Allocate memory on GPU for created pointer
	cudaMalloc((void**)&CUDA_car, group_count * sizeof(car*));

	for (auto group_index = 0; group_index < group_count; group_index++)
	{
		// Create pointer of type Car on GPU memory
		car * CUDA_car_child;

		// Allocate memory on GPU for created object
		cudaMalloc((void**)&CUDA_car_child, data_array_size);

		// Copy data of data array i-th element from CPU to GPU memory
		cudaMemcpy(CUDA_car_child, Cars_data[group_index], data_array_size, cudaMemcpyHostToDevice);
		// Copy pointer of data array i-th element from CPU to GPU memory (to Car array i-th place)
		cudaMemcpy(&CUDA_car[group_index], &CUDA_car_child, sizeof(car*), cudaMemcpyHostToDevice);
	}

	// Create pointer of type Car array on GPU memory for storing results
	car *CUDA_result;

	// Allocate memory on GPU for created pointer
	cudaMalloc((void**)&CUDA_result, data_array_size);

	// Run add function on GPU
	add <<<1, n>>> (CUDA_car, CUDA_result);

	// Wait for all CUDA threads to finish
	cudaDeviceSynchronize();

	// Copy data of results array back from GPU to CPU memory
	cudaMemcpy(Car_results, CUDA_result, data_array_size, cudaMemcpyDeviceToHost);

	// Free up no longer used memory on GPU
	for (auto i = 0; i < group_count; i++)
	{
		cudaFree(&CUDA_car[i]);
	}

	cudaFree(CUDA_car);
	cudaFree(CUDA_result);

	// Write results to file
	write_result();

	// Pause console window
	system("pause");
	return 0;
}
