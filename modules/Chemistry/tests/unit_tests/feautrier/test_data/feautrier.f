C*************************************************************************
C*************************************************************************
C ROUTINE IN WHICH THE EQUATION OF RAD TRANSFER IS SOLVED
C*************************************************************************
C*************************************************************************
      PROGRAM TRANSP
C*************************************************************************
C*************************************************************************
C  SOLVES THE RADIATIVE TRANSFER EQUATION WITH GIVEN SOURCE FUNCTION.
C  SOLVES FOR P-S WHERE P IS FEAUTRIERS P; I.E.
C
C    P = 0.5*(I(XMU)+I(-XMU))
C
C  IN THE PRESENCE OF VELOCITY FIELDS:
C
C    P = 0.5*(I(NY,XMU) + I(-NY,-XMU))
C

C  NDEP  : NUMBER OF DEPTH POINTS                               (IN)
C  S     : MONOCHROMATIC SOURCE FUNCTION                        (IN)
C  P     : MEAN BIDIRECTIONAL INTENSITY (CF. ABOVE)            (OUT)
C  TAUQ  : MONOCHROMATIC OPTICAL DEPTH                         (OUT)
C  DTAUQ : DTAUQ(K)=TAUQ(K)-TAUQ(K-1)                          (OUT)
C  A1    : 1/(DTAUQ(K+0.5)*DTAUQ(K))                           (OUT)
C  C1    : 1/(DTAUQ(K+0.5)*DTAUQ(K+1))                         (OUT)
C
C:

      IMPLICIT NONE
      
      integer i, k, mdep,ntrani
      PARAMETER(mdep=200)
      double precision a1(mdep),c1(mdep),dtauq(mdep),s(mdep)
      double precision p(mdep), IBC
C: Open ray file
      ntrani=0
      open(10,file='ray1.dat',status='old')
C read in background INTENSITY
      read(10,*) IBC
C read in text line
      read(10,*)
C read in source function and optical depth at each cell k
      do 10 i=1,1000,1
        read(10,*,end=20) k,s(i),dtauq(i)
        ntrani = ntrani+1
 10   continue
 20   close(10)

      write(*,*) 'Number of grid points is : ', ntrani
C Calculate element parts at gp=1 and gp = ntrani
      A1(1)=1.0d0/dtauq(1)
      A1(ntrani)=1.0d0/dtauq(ntrani)
C Calculate element parts a1 and c1 for gp=2 to ntrani-1
      DO 120 K=2,NTRANI-1,1

        A1(K)=2.0d0/ (DTAUQ(K)+DTAUQ(K+1)) / DTAUQ(K)
        C1(K)=2.0d0/ (DTAUQ(K)+DTAUQ(K+1)) / DTAUQ(K+1)

  120 CONTINUE
C Call matrix inversion solver
      CALL TRANF(a1,c1,s,dtauq,ibc,ntrani,p)
C write output to the output file
      open(11,file='intens_1.dat',status='unknown')
      do 130 i=1,ntrani
        write(11,*) s(i), dtauq(i), p(i)
 130  CONTINUE
      close(11)
      STOP
      END
C*************************************************************************
C*************************************************************************
C COMPUTATIONAL PHYSICS COURSE - SUBROUTINE TRANF
C
C*************************************************************************
C*************************************************************************
      SUBROUTINE TRANF(a1,c1,s,dtauq,ibc,ntrani,p)
C*************************************************************************
C*************************************************************************

      INTEGER k, MDEP
      PARAMETER(MDEP=200)    
      double precision a1(mdep),c1(mdep),dtauq(mdep),s(mdep)
      double precision p(mdep), IBC
      double precision bet, sp1(mdep), sp2(mdep), sp3(mdep), gam(mdep)   

C sp1, sp2, sp3 are the lower, main and upper diagonals
C dtauq(k=i+1) is the optical depth between k=i and K=i+1
C IBC is the input boundary condition
C P initially contains the suorce function.  The intensities are returned.

C Calculate sp1, sp2 and sp3

      do 10 k=2,ntrani-1
C Lower diagonal
       sp1(k) = -a1(k)
C main diagonal
       sp2(k) = 1 + a1(k) + c1(k)
C Upper diagonal
       sp3(k) = -c1(k)
C Remember that p holds the source function before entering the matrix
C solver.
       p(k) = s(k)
 10   continue

C Calculate boundary condition values of sp1, sp2 and sp3
C matrix elements Depth point = 1
      sp1(1) = 0.0d0
      sp2(1) = 1.0d0 + 2.0d0*a1(1) + 2.0d0*a1(1)*a1(1)
      sp3(1) = -2.0d0*a1(1)*a1(1)
C boundary conditions.
      FACT = 0.50d0*dtauq(2)
      P(1) = S(1) + IBC* DEXP(-dtauq(1))/FACT

C matrix elements Depth point = NTRANI
      sp1(ntrani) = -2.0d0*a1(ntrani)*a1(ntrani)
      sp2(ntrani) = 1.0d0 + 2.0d0*a1(ntrani) + 
     &              2.0d0*a1(ntrani)*a1(ntrani)
      sp3(ntrani) = 0.0d0
C boundary conditions.
      FACT = 0.50d0*dtauq(ntrani)
C By symmetry dtauq(1) = dtauq(ntrani+1)
      P(ntrani) = S(ntrani) + IBC* DEXP(-dtauq(1))/FACT
C Go into the matrix inversion routine

      do i=1,ntrani
        write(*,*) i,sp1(i),sp2(i),sp3(i),p(i)
      enddo

C Elimination step

      bet=sp2(1)
      p(1) = p(1) / bet

      do k=2, ntrani

        gam(k) = sp3(k-1)/bet
        bet = sp2(k) - (sp1(k)*gam(k))
        p(k) = (p(k) - sp1(k)*p(k-1)) / bet
        
      enddo

C Back substitution

      do k=ntrani-1,1,-1

        p(k) = p(k) - gam(k+1)*p(k+1)

      enddo

      RETURN
      END
