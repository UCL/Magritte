#include <mpi.h>
int main(){
int c;
char **v;
MPI_Init(&c, &v);
MPI_Finalize();
return(0);
}
