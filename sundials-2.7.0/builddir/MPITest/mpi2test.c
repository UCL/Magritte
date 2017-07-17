#include <mpi.h>
int main(){
int c;
char **v;
MPI_Comm C_comm;
MPI_Init(&c, &v);
C_comm = MPI_Comm_f2c((MPI_Fint) 1);
MPI_Finalize();
return(0);
}
