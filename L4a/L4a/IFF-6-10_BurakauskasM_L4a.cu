
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

struct car
{
	char manufacturer[max_string_length];
	char model[max_string_length];
	int year;
	double price;
};

car Cars_data[group_count][n];
car Car_results[n];

int Car_data_sizes[group_count];

void clear_results_file()
{
	ofstream file;
	file.open(results_filename, ofstream::out | ofstream::trunc);
	file.close();
}

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

__device__ char * custom_strcpy(char *destination, const char *source) {
	auto i = 0;

	do {
		destination[i] = source[i];
	} while (source[i++] != 0);

	return destination;
}

__device__ char * custom_strcat(char *destination, const char *source) {
	auto i = 0;

	while (destination[i] != 0)
		i++;

	custom_strcpy(destination + i, source);

	return destination;
}

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

	read_data();
	write_data();

	int data_array_size = n * sizeof(car);

	car **CUDA_car;
	cudaMalloc((void**)&CUDA_car, group_count * sizeof(car*));

	for (auto group_index = 0; group_index < group_count; group_index++)
	{
		car * CUDA_car_child;

		cudaMalloc((void**)&CUDA_car_child, data_array_size);

		cudaMemcpy(CUDA_car_child, Cars_data[group_index], data_array_size, cudaMemcpyHostToDevice);
		cudaMemcpy(&CUDA_car[group_index], &CUDA_car_child, sizeof(car*), cudaMemcpyHostToDevice);
	}

	car *CUDA_result;

	cudaMalloc((void**)&CUDA_result, data_array_size);

	add <<<1, n>>> (CUDA_car, CUDA_result);

	cudaDeviceSynchronize();

	cudaMemcpy(Car_results, CUDA_result, data_array_size, cudaMemcpyDeviceToHost);

	for (auto i = 0; i < group_count; i++)
	{
		cudaFree(&CUDA_car[i]);
	}

	cudaFree(CUDA_car);
	cudaFree(CUDA_result);

	write_result();

	system("pause");
	return 0;
}
