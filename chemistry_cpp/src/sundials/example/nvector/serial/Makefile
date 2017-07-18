# -----------------------------------------------------------------
# $Revision: 4958 $
# $Date: 2016-09-23 14:02:13 -0700 (Fri, 23 Sep 2016) $
# -----------------------------------------------------------------
# Programmer: Radu Serban @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see the LICENSE file.
# -----------------------------------------------------------------
# Makefile for  serial examples.
#
# This file is generated from a template using various variables
# set at configuration time. It can be used as a template for
# other user Makefiles.
#
# Note: if the solver was successfully configured with Blas/Lapack
# support, the Blas/Lapack libraries are specified through the 
# variable LIBRARIES_BL. Otherwise, this variable should contain
# an empty string. We include LIBRARIES_BL in the link line for
# all examples, whether they use the Lapack module or not, to
# address the case in which the SUNDIALS libraries are shared 
# objects. In that case, the solver library references Lapack 
# symbols which must be always resolved by linking against the
# Blas/Lapack libraries. If only static SUNDIALS libraries have 
# been built, it is not required to link the Blas/Lapack libraries
# for examples that do not use that module...
# -----------------------------------------------------------------

SHELL = sh

prefix       = /home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials
exec_prefix  = /home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials
includedir   = /home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/include
libdir       = /home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/lib

CPP      = /opt/intel/compilers_and_libraries_2017/linux/bin/intel64/icc
CPPFLAGS = -O3 -DNDEBUG
CC       = /opt/intel/compilers_and_libraries_2017/linux/bin/intel64/icc
CFLAGS   = -O3 -DNDEBUG
LDFLAGS  = 
LIBS     =  -lm /usr/lib/x86_64-linux-gnu/librt.so

LINKFLAGS = -Wl,-rpath,/home/frederik/Dropbox/Astro/3D-RT/chemistry_cpp/src/sundials/lib

INCLUDES = -I${includedir}
LIBRARIES = -lsundials_nvecserial -lsundials_nvecserial ${LIBS}
LIBRARIES_BL = 
LIBRARIES_SLUMT = 
LIBRARIES_KLU = 
LIBRARIES_PETSC = 

EXAMPLES =  test_nvector_serial 
EXAMPLES_DEPENDENCIES =  test_nvector sundials_nvector

OBJECTS = ${EXAMPLES:=.o}
OBJECTS_DEPENDENCIES = ${EXAMPLES_DEPENDENCIES:=.o}

# -----------------------------------------------------------------------------------------

.SUFFIXES : .o .c

.c.o :
	${CC} ${CPPFLAGS} ${CFLAGS} ${INCLUDES} -c $<

# -----------------------------------------------------------------------------------------

all: ${OBJECTS}
	@for i in ${EXAMPLES} ; do \
	  echo "${CC} -o $${i} $${i}.o ${OBJECTS_DEPENDENCIES} ${CFLAGS} ${LDFLAGS} ${INCLUDES} -L${libdir} ${LIBRARIES} ${LIBRARIES_BL} ${LIBRARIES_SLUMT} ${LIBRARIES_KLU} ${LIBRARIES_PETSC} ${LINKFLAGS}" ; \
	  ${CC} -o $${i} $${i}.o ${OBJECTS_DEPENDENCIES} ${CFLAGS} ${LDFLAGS} ${INCLUDES} -L${libdir} ${LIBRARIES} ${LIBRARIES_BL} ${LIBRARIES_SLUMT} ${LIBRARIES_KLU} ${LIBRARIES_PETSC} ${LINKFLAGS} ; \
	done

${OBJECTS}: ${OBJECTS_DEPENDENCIES}

clean:
	rm -f ${OBJECTS_DEPENDENCIES}
	rm -f ${OBJECTS}
	rm -f ${EXAMPLES}

# -----------------------------------------------------------------------------------------

