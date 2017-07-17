       INCLUDE "mpif.h"
       INTEGER IER
       CALL MPI_INIT(IER)
       CALL MPI_FINALIZE(IER)
       STOP
       END
