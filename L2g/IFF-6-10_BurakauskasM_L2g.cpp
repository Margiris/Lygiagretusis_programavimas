// Margiris Burakauskas, IFF-6/10
#include <iostream>
#include <mutex>
#include <thread>

using namespace std;

const int c = 10;
const int d = 100;

const int output_count_max = 15;
const int read_count_max = 2;

class monitor
{
    int c_;
    int d_;

    mutex mtx_;
    condition_variable cv_;

    bool program_finished_;

    int output_count_;
    int read_count_;

    public:
    monitor()
    {
        c_ = c;
        d_ = d;

        program_finished_ = false;

        output_count_ = 0;
        read_count_ = 0;
    }

    void change_c_d(const int thread_index)
    {
        unique_lock<mutex> lck(mtx_);
        
        while (!program_finished_)
        {
            cv_.wait(lck, [&] { return read_count_ >= read_count_max || program_finished_; });
            
            c_ += thread_index;
            d_ -= thread_index;

            read_count_ = 0;

            cv_.notify_all();
        }

    }

    void read_c_d(const int thread_index)
    {
        auto prev_c = 0;
        auto prev_d = 0;
       
        unique_lock<mutex> lck(mtx_);

        while (!program_finished_)
        {
            cv_.wait(lck, [&] { return (prev_c != c_ && prev_d != d_) || program_finished_; });
            
            if (output_count_ < output_count_max)
            {
                cout << "Gija " << thread_index << ", c: " << c_ << ", d: " << d_ << "\n";

                output_count_++;
                read_count_++;

                prev_c = c_;
                prev_d = d_;
            }
            else
            {
                program_finished_ = true;
            }

            cv_.notify_all();
        }
    }
};

int main()
{
    monitor monitor;

    thread threads[5];

    threads[0] = thread(&monitor::change_c_d, &monitor, 1);
    threads[1] = thread(&monitor::change_c_d, &monitor, 2);

    threads[2] = thread(&monitor::read_c_d, &monitor, 3);
    threads[3] = thread(&monitor::read_c_d, &monitor, 4);
    threads[4] = thread(&monitor::read_c_d, &monitor, 5);

    for (auto& thread : threads)
    {
        thread.join();
    }

    system("pause");
    return 0;
}