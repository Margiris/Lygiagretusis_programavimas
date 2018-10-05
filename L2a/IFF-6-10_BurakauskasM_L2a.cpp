#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <iomanip>
#include <vector>
#include <map>
#include <mutex>
#include <sstream>
#include <iterator>

using namespace std;

const string data_filename = R"(IFF-6-10_BurakauskasM_L2a_dat.txt)";
const string results_filename = R"(IFF-6-10_BurakauskasM_L2a_rez.txt)";
const string delimiter = ";";

struct teacher
{
    string name;
    int working_year{};
    double average_salary_per_hour{};
};

class sorting_structure
{
    public:
    sorting_structure(int sorting_field, int count)
        : sortingField(sorting_field),
          count(count)
    {
    }

    private:
    int sortingField;
    int count;
};


vector<teacher> read_data()
{
    vector<teacher> data;

    string line;

    ifstream data_file(data_filename);

    while (getline(data_file, line))
    {
        vector<string> values;
        size_t pos;

        while ((pos = line.find(delimiter)) != string::npos)
        {
            values.push_back(line.substr(0, pos));
            line.erase(0, pos + 1);
        }

        if (!line.empty())
        {
            values.push_back(line);
        }

        stringstream line_stream;
        copy(values.begin(), values.end(), ostream_iterator<string>(line_stream, " "));

        switch (values.size())
        {
            case 2:

                break;
            case 4:
            {
                string faculty, name;
                int year;
                double average_salary_per_hour;

                line_stream >> faculty >> name >> year >> average_salary_per_hour;

                data.push_back(teacher{name, year, average_salary_per_hour});
                break;
            }
            default:
                cout << "Wrong data, skipping line." << endl;
                break;
        }
    }

    data_file.close();

    return data;
}

int main()
{
    read_data();
    system("pause");
}
