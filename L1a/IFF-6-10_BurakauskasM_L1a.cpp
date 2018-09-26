// ReSharper disable CommentTypo
/*
1. Tokia, kokia uzrasyti
2. Atsitiktine
3. Atsitiktini skaiciu
4. Tokia, kokia surasyti duomenu masyve
*/

#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <iomanip>
#include <vector>
#include <map>
#include <mutex>

using namespace std;

const string data_filename = R"(IFF-6-10_BurakauskasM_L1a_dat.txt)";
const string results_filename = R"(IFF-6-10_BurakauskasM_L1a_rez.txt)";

// Fakultetu enumeratorius
enum faculties
{
    informatics,
    economics,
    mechanics,
    chemistry,
    mathematics
};

// Susieja duomenu failo fakultetus su enumeratoriaus reiksmemis
static map<string, int> faculties_map;

// Suranda enumeratoriaus zemelapyje rakta pagal reiksme
template <typename K, typename V>
string find_by_value(map<K, V> map_of_elemen, V value)
{
    auto it = map_of_elemen.begin();
    // Iterate through the map
    while (it != map_of_elemen.end())
    {
        // Check if value of this entry matches with given value
        if (it->second == value)
        {
            return (it->first);
        }
        ++it;
    }
    return "Not found";
}

// Inicializuoja fakultetu enumeratoriaus zemelapi
void initialize_faculties_map()
{
    faculties_map["Informatikos"] = informatics;
    faculties_map["Ekonomikos"] = economics;
    faculties_map["Mechanikos"] = mechanics;
    faculties_map["Chemijos"] = chemistry;
    faculties_map["Matematikos"] = mathematics;
}

// Destytojo duomenu struktura
struct teacher
{
    string name;
    int working_year{};
    double average_salary_per_hour{};
};

// Destytojo duom. struktura rezultatams saugoti
struct teacher_results
{
    string name;
    int working_year{};
    double average_salary_per_hour{};
    string thread_name;
    int index{};
};

// Bendras visoms gijoms rezultatu masyvas
teacher_results results[100];

// Bendras visoms gijoms paskutinio elemento rezultatu masyve indeksas
int last_element_index;

// Perskaito destytoju duomenis i vektorius o pastaruosius issaugo ir grazina viename vektoriuje
vector<vector<teacher>> read_data()
{
    vector<vector<teacher>> data(faculties_map.size());

    string faculty;
    string name;
    int year;
    double average_grade;

    ifstream data_file(data_filename);

    while (data_file >> faculty >> name >> year >> average_grade)
    {
        teacher tchr;

        try
        {
            tchr.name = name;
            tchr.working_year = year;
            tchr.average_salary_per_hour = average_grade;
            data[faculties_map.at(faculty)].push_back(tchr);
        }
        catch (exception& e)
        {
            cout << "Wrong data, skipping line. (exception: " << e.what() << ")." << endl;
        }
    }

    data_file.close();

    return data;
}

// Suraso pradinius duomenis i faila lentele
void write_data(vector<vector<teacher>> data)
{
    ofstream results_file(results_filename);

    for (auto i = 0; i < int(data.size()); i++)
    {
        results_file << "*** " << find_by_value(faculties_map, i) << " ***" << endl;
        results_file << left << setw(4) << "Nr." << setw(20) << "Vardas" << setw(8) << "Metai" << setw(9) << "Alga / h"
            << endl;

        for (auto o = 0; o < int(data[i].size()); o++)
        {
            results_file << setw(4) << right << to_string(o + 1) + ") " << setw(20) << left << data[i][o].name <<
                setw(8) << data[i][o].working_year << setw(9) << data[i][o].average_salary_per_hour << endl;
        }

        results_file << endl;
    }

    results_file.close();
}

// Suraso rezultatu vektoriu i faila lentele
void write_results()
{
    ofstream results_file(results_filename, ios::app);

    results_file << left << setw(3) << "Nr" << setw(11) << "Gija" << setw(10) << "Eiles nr." << setw(20) << "Vardas" <<
        setw(8) << "Metai"
        << setw(9) << "Alga / h" << endl;

    auto i = 0;
    while (!results[i].thread_name.empty())
    {
        results_file << setw(2) << right << i + 1 << " " << setw(10) << left << results[i].thread_name << " " <<
            setw(10) << results[i].index << setw(20) << results[i].name << setw(8) <<
            results[i].working_year << setw(9) << results[i].average_salary_per_hour << endl;
        i++;
    }

    results_file.close();
}

// Suraso duomenis is paduoto vektoriaus i bendra rezultatu vektoriu
void fill_array(vector<teacher> faculty_teachers, const int thread_id)
{
    for (auto i = 0; i < int(faculty_teachers.size()); i++)
    {
        teacher_results s;

        s.name = faculty_teachers[i].name;
        s.working_year = faculty_teachers[i].working_year;
        s.average_salary_per_hour = faculty_teachers[i].average_salary_per_hour;
        s.thread_name = "gija_nr_" + to_string(thread_id);
        s.index = i + 1;

        results[last_element_index++] = s;

        // Papildomas darbas gijoms, kad labiau issimaisytu rasydamos rezultatus
        for (auto o = 0; o < 1000; o++)
        {
            // ReSharper disable once CppDeclaratorNeverUsed
            auto k = sqrt(o);
            cout << i;
        }
    }
}

int main()
{
    initialize_faculties_map();

    // ReSharper disable once CppUseAuto
    const vector<vector<teacher>> data = read_data();

    write_data(data);

    const auto threads_arr = new thread[data.size()];

    // Sukuria gijas
    for (auto i = 0; i < int(data.size()); i++)
    {
        threads_arr[i] = thread(fill_array, data[i], i);
    }
    // Palaukia visu giju darbo pabaigos
    for (auto i = 0; i < int(data.size()); i++)
    {
        threads_arr[i].join();
    }

    write_results();

    delete[](threads_arr);

    system("pause");

    return 0;
}
