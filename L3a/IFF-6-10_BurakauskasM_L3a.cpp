#include <mpi.h>
#include <cstdlib>
#include <cstdio>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        char hello_str[] = "Hello World";
        MPI_Send(hello_str, _countof(hello_str), MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1)
    {
        char hello_str[12];
        MPI_Recv(hello_str, _countof(hello_str), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank 1 received string %s from Rank 0\n", hello_str);
    }

    MPI_Finalize();
    return 0;
}
