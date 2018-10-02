// ReSharper disable CommentTypo
/*
1. Atsitiktine (visos beveik vienu metu)
2. Atsitiktine
3. Vienos dali
4. Atsitiktine
*/
// ReSharper restore CommentTypo

#include <fstream>
#include <string>
#include <iomanip>
#include <omp.h>

using namespace std;

// Number of lines read from data file and threads created
const int n = 30;

// Names of data and results files
const string data_filename = R"(IFF-6-10_BurakauskasM_L1b_dat.txt)";
const string results_filename = R"(IFF-6-10_BurakauskasM_L1b_rez.txt)";

// Structure for saving results
struct results
{
    int id{};
    string a_string;
    int an_integer{};
    double a_double{};
};

// Reads data from file to S, I and D arrays
// ReSharper disable CppInconsistentNaming
void read_data(string S[], int I[], double D[])
// ReSharper restore CppInconsistentNaming
{
    ifstream data_file(data_filename);

    for (auto i = 0; i < n; i++)
    {
        data_file >> S[i] >> I[i] >> D[i];
    }

    data_file.close();
}


// Writes data from S, I and D arrays to results file
// ReSharper disable CppInconsistentNaming
void write_data(string S[], int I[], double D[])
// ReSharper restore CppInconsistentNaming
{
    ofstream results_file(results_filename);

    results_file << left << setw(4) << "No." << setw(20) << "string" << setw(8) << "int" << setw(8) << "double" << endl;
    results_file << string(40, '-') << endl;

    for (auto i = 0; i < n; i++)
    {
        results_file << setw(4) << right << to_string(i + 1) + ") " << setw(20) << left << S[i] << setw(8) << I[i] <<
            setw(8) << D[i] << endl;
    }

    results_file << string(40, '-') << endl;

    results_file.close();
}

// Writes data from results array P to results file
// ReSharper disable CppInconsistentNaming
void write_results(results P[])
// ReSharper restore CppInconsistentNaming
{
    ofstream results_file(results_filename, ios::app);

    results_file << left << setw(4) << "No." << setw(7) << "Thread" << setw(20) << "string" << setw(8) << "int" <<
        setw(8) << "double" << endl;
    results_file << string(45, '-') << endl;

    for (auto i = 0; i < n; i++)
    {
        results_file << setw(4) << right << to_string(i + 1) + ") " << setw(6) << P[i].id << " " << setw(20) << left <<
            P[i].a_string << setw(8) << P[i].an_integer << setw(8) << P[i].a_double << endl;
    }

    results_file << string(45, '-') << endl;

    results_file.close();
}

int main()
{
    // ReSharper disable CppInconsistentNaming
    string S[n];
    int I[n];
    double D[n];
    results P[n];
    // ReSharper restore CppInconsistentNaming

    read_data(S, I, D);
    write_data(S, I, D);

    auto i = 0;

    // Creates n threads which all write their data to the same array 
    #pragma omp parallel num_threads(n)
    {
        const auto thread_number = omp_get_thread_num();

        P[i++] = results{thread_number + 1, S[thread_number], I[thread_number], D[thread_number]};
    }

    write_results(P);

    return 0;
}
