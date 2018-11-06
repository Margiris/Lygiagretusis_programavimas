#include <thread>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>
#include <conio.h>

using namespace std;

bool end_program = false;

struct monitorius
{
    int io_counter = 0;
    int read_counter_c = 0;
    int read_counter_d = 0;
    condition_variable cv;
    mutex mtx;
    int c = 10;
    int d = 100;

    monitorius()
    {
        c = 10;
        d = 100;
        io_counter = 0;
        read_counter_c = 0;
        read_counter_d = 0;
    }

    void change_c(const int i)
    {
        while (!end_program)
        {
            unique_lock<mutex> lck(mtx);
            cv.wait(lck, [&] { return read_counter_c > 1; });
            if (!end_program)
            {
                c += i;
                read_counter_c = 0;
                cv.notify_all();
            }
        }
    }

    void change_d(const int i)
    {
        while (!end_program)
        {
            unique_lock<mutex> lck(mtx);
            cv.wait(lck, [&] { return read_counter_d > 1; });
            if (!end_program)
            {
                d -= i;
                read_counter_d = 0;
                cv.notify_all();
            }
        }
    }

    void read_cd(const int i)
    {
        auto old_c = -1;
        auto old_d = -1;
        while (!end_program)
        {
            unique_lock<mutex> lck(mtx);
            cv.wait(lck, [&] { return old_c != c && old_d != d; });
            if (!end_program)
            {
                stringstream str;
                str << i << ": c:" << c << " d: " << d << endl;
                cout << str.str();
                read_counter_c++;
                read_counter_d++;
                io_counter++;
                if (io_counter >= 15)
                    end_program = true;
                cv.notify_all();
                old_c = c;
                old_d = d;
            }
        }
    }
};

int main()
{
    monitorius m;
    vector<thread> my_threads;

    my_threads.reserve(5);

    my_threads.emplace_back([&] { m.change_c(1); });
    my_threads.emplace_back([&] { m.change_d(2); });
    my_threads.emplace_back([&] { m.read_cd(3); });
    my_threads.emplace_back([&] { m.read_cd(4); });
    my_threads.emplace_back([&] { m.read_cd(5); });

    for (size_t i = 0; i < 1; i++)
        my_threads[i].join();

    _getch();
}
