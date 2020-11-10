MODULE Routines
  IMPLICIT NONE
  CONTAINS

FUNCTION delta2(No_data,z,p,del,alpha,z_data,sigma_data)
 IMPLICIT NONE

  INTEGER,INTENT(IN) :: No_data
  REAL*8,INTENT(IN),DIMENSION(No_data) :: z_data, sigma_data
  REAL*8,INTENT(IN) :: z,p,del,alpha
!  REAL*8,PARAMETER :: del = 1.0D0			!type 1: power 1.5 delta 0.3 and const. 1.9  ---- type 2 delta 0.1 const 1.8
  REAL*8 :: delta2
  INTEGER :: i

  delta2 = 0.0D0
  DO i = 1,No_data	
	delta2 = delta2 + (((1.0D0)/((sigma_data(i))**2))*(EXP(-1.0D0*(( (z-z_data(i))/(del))**2))))	!for type 1 -- 2 power is 1.5
 END DO
  delta2 = (alpha)*((1.0D0/delta2)**(1.0D0/p))

 RETURN
END FUNCTION delta2

FUNCTION delta(No_data,z,z_data,sigma_data)
IMPLICIT NONE
	INTEGER,INTENT(IN) :: No_data
	REAL*8,INTENT(IN),DIMENSION(No_data) :: z_data, sigma_data
	REAL*8,INTENT(IN) :: z!,del,p
	REAL*8,PARAMETER :: del = 0.1D0,p = 1.5			!type 1: power 1.5 delta 0.3 and const. 1.9  ---- type 2 delta 0.1 const 1.8
	REAL*8 :: delta
	INTEGER :: i
	delta = 0.0D0
	DO i = 1,No_data	
	delta = delta + EXP(-1.0D0*(( ((z-z_data(i))*(sigma_data(i))**(p))/(del))**2))	!for type 1 -- 2 power is 1.5
	END DO
    delta = 0.2D1-(delta/580.0D0)-0.2
RETURN
END FUNCTION delta


!==========================================================================
!		DM_flat
!
!	This function gives the	Distance modulus for F-LCDM univers.
!	inputs: Omega matter: Om	Hubble const.: H0	Redshift: X2
!
!==========================================================================
FUNCTION DM_flat (X2, H0, Om)
  IMPLICIT NONE
  REAL*8 :: DM_flat
  INTEGER :: i,n
  REAL*8 :: X1, H_X1, sum2
  REAL*8, INTENT (IN) :: X2, H0, Om

  sum2 = 0.0D0
  X1 = 0.0D0
  n = 100
  H_X1=(X2-X1)/float(n)
  DO i = 1, n
     sum2 = sum2 +((1.0D0)/(sqrt((Om)*((X1+1.0D0)**3)+1.0D0-Om)))
     X1=X1+H_X1
  END DO
  sum2=(sum2)*((1.0D0+X2)*(3.0D5)*(H_X1/H0))
  DM_flat = 5.0D0*LOG10(sum2) + 25.0D0
RETURN
END FUNCTION DM_flat




!		Hubble parameter
FUNCTION Hubfunc (X2, Omega)
	IMPLICIT NONE

	REAL*8 :: Hubfunc
	REAL*8, INTENT (IN) :: X2, Omega
!	Hubfunc = SQRT((Omega)*((1.0+X2)**3) + 1.0-(Omega))
	Hubfunc = SQRT((Omega)*((X2+1.0D0)**3)+(1.0D0-Omega)*(Kink_rho (X2)))
RETURN
END FUNCTION Hubfunc

!


!		Hubble parameter
FUNCTION Hubfunc_lcdm (X2, Omega)
	IMPLICIT NONE

	REAL*8 :: Hubfunc_lcdm
	REAL*8, INTENT (IN) :: X2, Omega
	Hubfunc_lcdm = SQRT((Omega)*((1.0+X2)**3) + 1.0-(Omega))
RETURN
END FUNCTION Hubfunc_lcdm
!
!
FUNCTION w(z)
IMPLICIT NONE
	REAL*8,PARAMETER :: w0 = -1.0D0,wm = -0.5D0,zt = 0.5D0,delta_t = 0.5D-1
	REAL*8 :: w
	REAL*8,INTENT(IN) :: z
	w = (3.0D0/(1.0D0+z))*(1.0D0+w0 + ((wm-w0)*((1.0D0+EXP(1.0D0/(delta_t*(1.0D0+zt))))/(1.0D0-EXP(1.0D0/delta_t)))*(1.0D0- &
	((EXP(1.0D0/delta_t)+EXP(1.0D0/(delta_t*(1.0D0+zt))))/(EXP(1.0D0/(delta_t*(1.0D0+z)))+EXP(1.0D0/(delta_t*(1.0D0+zt))))))))
RETURN
END FUNCTION w

FUNCTION Kink_rho (b)
IMPLICIT NONE
    real*8, parameter:: a=0.0D0
    integer, parameter :: n = 100 
	INTEGER :: i
	REAL*8 :: h, sum, x,rho
	REAL*8,INTENT(IN) :: b
	REAL*8 :: Kink_rho

	h = (b-a)/real(n)  
	rho = 0.0D0
	rho = 0.5D0*(w(a) + w(b))     
	DO  i=1,n-1
		x = a + i*h
		rho = rho + w(x)   
	END DO

Kink_rho = EXP(h*rho)
RETURN
END FUNCTION Kink_rho

FUNCTION Kink_mu (X2, H0)
 IMPLICIT NONE

 INTEGER :: i,n

 REAL*8, PARAMETER :: Om = 0.3D0
 REAL*8 :: X1, H_X1, sum2,rho
 REAL*8, INTENT (IN) :: X2, H0
 REAL*8 :: Kink_mu

 sum2 = 0.0D0
 X1 = 0.0D0
 n = 100
 H_X1=(X2-X1)/float(n)

  DO i = 1, n
    sum2 = sum2 +((1.0D0)/(sqrt((Om)*((X1+1.0D0)**3)+(1.0D0-Om)*(Kink_rho (X1)))))
    X1=X1+H_X1
  END DO

 sum2=(sum2)*((1.0D0+X2)*(3.0D5)*(H_X1/H0))
 Kink_mu = 5.0D0*LOG10(sum2) + 25.0D0

RETURN
END FUNCTION Kink_mu

SUBROUTINE LCDM_qOm (No_bins,z,q,Om,Omega,H)
IMPLICIT NONE

 INTEGER,INTENT(IN) :: No_bins
 REAL*8,DIMENSION(No_bins),INTENT(IN) :: z
 REAL*8,DIMENSION(No_bins),INTENT(OUT) :: q,Om,H
 REAL*8,INTENT(IN) :: Omega 
 REAL*8 :: x
 INTEGER :: j 


 DO j=2,(No_bins-1)
   x = z(j)
   H(j) = Hubfunc_lcdm (x, Omega)
 END DO


	DO j=3,(No_bins-2)
		q(j) = (( (1+z(j)) / (H(j))) * ((H(j+1)-H(j-1))/(z(j+1)-z(j-1)))) - 1.0D0
	END DO
	DO j=3,(No_bins-2)
		Om(j) = (((H(j+1))**2)-((H(j-1))**2))/(((1.0D0+z(j+1))**3)-((1.0D0+z(j-1))**3)) 
	END DO

RETURN
END SUBROUTINE LCDM_qOm

!
!	This routine computes deceleration parameter for Kink universe
!
SUBROUTINE Kink_qOm (No_bins,z,q,Om,Omega,H)
IMPLICIT NONE

 INTEGER,INTENT(IN) :: No_bins
 REAL*8,DIMENSION(No_bins),INTENT(IN) :: z
 REAL*8,DIMENSION(No_bins),INTENT(OUT) :: q,Om,H
 REAL*8,INTENT(IN) :: Omega 
 REAL*8 :: x
 INTEGER :: j 


 DO j=2,(No_bins-1)
   x = z(j)
   H(j) = Hubfunc (x, Omega)
 END DO


	DO j=3,(No_bins-2)
		q(j) = (( (1+z(j)) / (H(j))) * ((H(j+1)-H(j-1))/(z(j+1)-z(j-1)))) - 1.0D0
	END DO
	DO j=3,(No_bins-2)
		Om(j) = (((H(j+1))**2)-((H(j-1))**2))/(((1.0D0+z(j+1))**3)-((1.0D0+z(j-1))**3)) 
	END DO

RETURN
END SUBROUTINE Kink_qOm
!
!
!	Routine to calculate the residual square between DM_data and Kink model
SUBROUTINE LF_Kinkmu (No_data,chi_min,z_data, DM_data,sigma_data)
	IMPLICIT NONE

	INTEGER :: i,j,t
	REAL*8 :: x, chi, H0
	INTEGER,INTENT(IN) :: No_data
	REAL*8,INTENT(OUT),DIMENSION(3) :: chi_min
	REAL*8,INTENT(IN),DIMENSION(No_data) :: z_data,sigma_data
	REAL*8,INTENT(IN),DIMENSION(3,No_data) :: DM_data
	chi_min = 10.0D5

DO t = 1,3
	H0 = 68.0D0
	DO j = 1,4
		chi = 0.0D0

		DO i = 1, No_data
			x = z_data(i)
			chi = chi + ((Kink_mu (x, H0) - (DM_data(t,i)))/(sigma_data(i)))**2
		END DO

		IF (chi .LE. chi_min(t)) THEN
		chi_min(t) = chi
		END IF

	H0 = H0 + 1.0D0
	END DO
END DO

RETURN
END SUBROUTINE LF_Kinkmu



!
!	Routine to calculate the residual square between DM_data and Kink model
SUBROUTINE LF_muLCDM (No_data,chi_min,H0_best,Om,z_data,sigma_data)
	IMPLICIT NONE

	INTEGER :: i,j
	REAL*8 :: x, chi, H0
	INTEGER,INTENT(IN) :: No_data
	REAL*8,INTENT(OUT) :: chi_min
    real*8,intent(in) :: H0_best,Om
	REAL*8,INTENT(IN),DIMENSION(No_data) :: z_data,sigma_data

	chi_min = 10.0D5
!	H0 = 68.0D0
!	DO j = 1, 10
	H0 = H0_best
		chi = 0.0D0

		DO i = 1, No_data
			x = z_data(i)
			chi = chi + ((Kink_mu (x, H0) - (  DM_flat (x, H0, Om)  ))/(sigma_data(i)))**2
		END DO

		IF (chi .LE. chi_min) THEN
!			H0_best = H0
			chi_min = chi
		END IF

!	H0 = H0 + 0.5D0
!	END DO
RETURN
END SUBROUTINE LF_muLCDM

!
SUBROUTINE Lf_OmQ (No_bins,Om,q,chi,chi2,LF_q_sum,Om_true,q_true,sigma_data)
	IMPLICIT NONE
	INTEGER,INTENT(IN) :: No_bins
	REAL*8,INTENT(IN),DIMENSION(3,No_bins) :: Om,q
	REAL*8,INTENT(IN),DIMENSION(No_bins) ::sigma_data
	REAL*8,DIMENSION(No_bins) :: q_sum
	REAL*8,INTENT(OUT),DIMENSION(3) :: chi,chi2
	REAL*8,INTENT(OUT) :: LF_q_sum
	INTEGER :: i,t
	REAL*8,INTENT(IN),DIMENSION(No_bins) :: Om_true,q_true

	chi = 0.0D0
	chi2 = 0.0D0

DO t =1,3
	DO i = 3, (No_bins-2)
		chi(t) = chi(t) +  ( ((Om(t,i)) - (Om_true(i)))/(sigma_data(i))   )**2
	END DO
	DO i = 3, (No_bins-2)
		chi2(t) = chi2(t) + (((q(t,i)) - (q_true(i)))/(sigma_data(i)) )**2  
	END DO
END DO

	LF_q_sum = 0D0
	q_sum = SUM(q,DIM = 1)
	q_sum = (q_sum)/(3.0D0)
	DO i = 3, (No_bins-2)
		LF_q_sum = LF_q_sum + (((q_sum(i)) - (q_true(i)))/(sigma_data(i)) )**2  
	END DO


RETURN
END SUBROUTINE Lf_OmQ

!LF_h_sum,LF_q_sum
SUBROUTINE Lf_hubble (No_bins,H,z,chi,Omega,LF_h_sum,sigma_data)
	IMPLICIT NONE
	INTEGER,INTENT(IN) :: No_bins
	REAL*8,INTENT(IN),DIMENSION(No_bins) :: z,sigma_data
	REAL*8,INTENT(IN),DIMENSION(3,No_bins) :: H
	REAL*8,INTENT(OUT),DIMENSION(3) :: chi
	REAL*8,INTENT(OUT) :: LF_h_sum
	REAL*8,INTENT(IN) :: Omega
	REAL*8,DIMENSION(No_bins) :: H_sum
	REAL*8 :: x
	INTEGER :: i,j

	chi = 0.0D0
DO j = 1,3

	DO i = 2, (No_bins-1)
		x = z(i)
		chi(j) = chi(j) +  (( (H(j,i)) - (Hubfunc(x,Omega))  )/(sigma_data(i)) )**2
	END DO
END DO

    H_sum = SUM(H, DIM = 1)
	H_sum = (H_sum)/(3.0D0)
    LF_h_sum = 0D0
	DO i = 2, (No_bins-1)
		x = z(i)
		LF_h_sum = LF_h_sum +  (( (H_sum(i)) - (Hubfunc(x,Omega))  )/(sigma_data(i)) )**2
	END DO

RETURN
END SUBROUTINE Lf_hubble

!
SUBROUTINE Reconstruction (No_bins,DM,z,H,q)
 IMPLICIT NONE

  INTEGER :: j
  REAL*8 :: H0
  INTEGER,INTENT(IN) :: No_bins
  REAL*8,INTENT(IN),DIMENSION(No_bins):: z
  REAL*8,INTENT(IN),DIMENSION(No_bins):: DM
  REAL*8,DIMENSION(No_bins):: L
  REAL*8,INTENT(OUT),DIMENSION(No_bins):: H,q!,Om

  H = 0.0D0;q = 0.0D0

  DO j = 1, No_bins
	L (j) = (1.0D0/3.0D10)*((10.0D0**((DM (j))/(5.0D0)))/(1.0D0 + z(j))) ! in Mpc.s/Km
  END DO

  H (1) = (z(2)-z(1))/(L(2)-L(1))
  DO j = 2, (No_bins-1)
	H (j) = (z(j+1)-z(j-1))/((L(j+1)-L(j-1)))
  END DO
  H0 = ((-1.0*z(2))*((H(3)-H(2))/(z(3)-z(2))))+H(2)! By extrapolation I find H(z = 0) and then devide all values by that to get h(z) = H(z)/H(0)

!	dividing H(z) by H(z = 0)
  DO j = 2, (No_bins-1)
	H (j) = H (j)/H0
  END DO



  DO j=3,(No_bins-2)
	q(j) = (( (1+z(j)) / (H(j))) * ((H(j+1)-H(j-1))/(z(j+1)-z(j-1)))) - 1.0D0
  END DO

!  DO j=3,(No_bins-2)
!	Om(j) = (((H(j+1))**2)-((H(j-1))**2))/(((1.0D0+z(j+1))**3)-((1.0D0+z(j-1))**3)) 
!  END DO


RETURN
END SUBROUTINE Reconstruction

!
!
!
SUBROUTINE interpolate(No_data,No_bins,z,X,z_data,Y)
  IMPLICIT NONE

  INTEGER :: i,j
  INTEGER,INTENT(IN) :: No_data,No_bins
  REAL*8,INTENT(IN),DIMENSION(No_bins) :: z,X
  REAL*8,INTENT(IN),DIMENSION(No_data) :: z_data
  REAL*8,INTENT(OUT),DIMENSION(No_data) :: Y

 do j=1,No_bins
	   do i = 1, No_data
		IF (z_data(i) .LE. z(j+1) .AND. z_data(i) .GE. z(j)) Y(i) = ((z_data(i)-z(j))*((X(j+1)-X(j))/(z(j+1)-z(j))))+X(j)
       end do
 end do
RETURN
END SUBROUTINE interpolate



SUBROUTINE interpolate2(No_data,No_bins,z,X,z_data,Y)
  IMPLICIT NONE

  INTEGER :: i,j,t
  INTEGER,INTENT(IN) :: No_data,No_bins
  REAL*8,INTENT(IN),DIMENSION(No_bins) :: z
  REAL*8,INTENT(IN),DIMENSION(3,No_bins) :: X
  REAL*8,INTENT(IN),DIMENSION(No_data) :: z_data
  REAL*8,INTENT(OUT),DIMENSION(3,No_data) :: Y

do t = 1, 3

 do j=1,No_bins
	   do i = 1, No_data
		IF (z_data(i) .LE. z(j+1) .AND. z_data(i) .GE. z(j)) Y(t,i) = ((z_data(i)-z(j))*((X(t,j+1)-X(t,j))/(z(j+1)-z(j))))+X(t,j)
       end do
 end do

end do

RETURN
END SUBROUTINE interpolate2


! to sort data sort p,... based on p>q or q>p
Subroutine Order(p,q,p1,q1,p2,q2)
implicit none
real*8 p,q,temp,p1,q1,p2,q2
  if (p>q) then
    temp=p
    p=q
    q=temp

    temp=p1
    p1=q1
    q1=temp

    temp=p2
    p2=q2
    q2=temp
  end if
  return
end Subroutine Order

Subroutine Bubble(A,B,C, n)
implicit none

real*8 A(1:n),B(1:n),C(1:n)
integer n,i,j

  do i=1, n
    do j=n, i+1, -1
      call Order(A(j-1), A(j),B(j-1), B(j),C(j-1), C(j))
    end do
  end do
  return
end Subroutine Bubble





!	Routine to calculate the residual square between DM_data and function DM_flat
SUBROUTINE Chi_flat (No_data,chi_min,H0_best,z_data, DM_data, sigma_data,Om)
	IMPLICIT NONE

	INTEGER :: i,j
	REAL*8 :: x, chi, H0
	INTEGER,INTENT(IN) :: No_data
	REAL*8,INTENT(OUT) :: chi_min,H0_best
	REAL*8,INTENT(IN) :: Om
	REAL*8,INTENT(IN),DIMENSION(No_data) :: z_data, DM_data, sigma_data

	chi_min = 10.0D5
	H0 = 65.0D0
	DO j = 1, 10
		chi = 0.0D0

		DO i = 1, No_data
			x = z_data(i)
			chi = chi + (( DM_flat (x,H0,Om) - (DM_data(i))  ) / (sigma_data(i)) )**2
		END DO

		IF (chi .LE. chi_min) THEN
			H0_best = H0
			chi_min = chi
		END IF

	H0 = H0 + 1.0D0
	END DO
RETURN
END SUBROUTINE Chi_flat

!==========================================================================
!				Chi_LCDM
!
!	This routine calculate chi-square between the function DM_flat and data DM_data
!	DM_flat is distance modulus in flat LCDM universe
!
!	inputs: DM_data,z_data,sigma_data,No_data
!	outputs: chi_min,H0 best, Om best
!
!==========================================================================
SUBROUTINE Chi_LCDM (No_data,z_data, DM_data, sigma_data,chi_min,H0_best,Om_best)
  IMPLICIT NONE

  INTEGER :: m,n,i
  INTEGER,INTENT(IN) :: No_data
  REAL*8,INTENT(IN),DIMENSION(No_data) :: z_data, DM_data, sigma_data
  REAL*8,INTENT(OUT) :: chi_min,H0_best,Om_best
  REAL*8 :: H01,Om1,chi2,z1

  chi_min = 10.0D5
  H01 = 68.0D0
  DO n = 1, 100
	 Om1 = 0.1D-10
	 DO m = 1,100
	     chi2 = 0.0D0

	     DO i = 1, No_data
	       z1 = z_data(i)
	       chi2 = chi2 + (((  DM_flat (z1,H01,Om1)) - (DM_data(i))  ) / (sigma_data(i)) )**2
	     END DO

	    IF (chi2 .LE. chi_min) THEN
	    	chi_min = chi2
	    	Om_best = Om1
			H0_best = H01
	    END IF

	 Om1 = Om1 + 0.01D0
	 END DO
  H01 = H01 + 0.1D0
  END DO
RETURN
END SUBROUTINE Chi_LCDM

!==========================================================================
!		Initial_guess
!
!	This routine 1- defines the grids, given binsize from the min - max of redshifts (data)
!				 2- based on F-LCDM universe (DM_flat function) initialize the smoothing method
!
!	inputs: No_data, No_bins	z_data: redshift of data	binsize
!			H0: hubble parameter	Om: matter density
!
!	outputs: DM_guess_data/bins : distance modulus on data points/grids
!==========================================================================
SUBROUTINE Initial_guess (No_data,No_bins,z_data,H0,binsize,Om,DM_guess_data,DM_guess_bins)
 IMPLICIT NONE

  INTEGER :: j,i
  INTEGER,INTENT(IN) :: No_data,No_bins
  REAL*8,INTENT(OUT),DIMENSION(No_data)  :: DM_guess_data
  REAL*8,INTENT(OUT),DIMENSION(No_bins) :: DM_guess_bins
  REAL*8,INTENT(IN),DIMENSION(No_data) :: z_data
  REAL*8,INTENT(IN) :: H0,binsize,Om
  REAL*8 :: z1,Mu

  DM_guess_bins = 0.0D0
  DM_guess_data = 0.0D0

  z1 = MINVAL(z_data)-(1.5D0*binsize)
  DO j = 1, No_bins
	DM_guess_bins(j) = DM_flat (z1, H0, Om)
	z1 = z1 + binsize
  END DO
!
!
  DO i = 1, No_data
   z1 = z_data(i)
   DM_guess_data (i) = DM_flat (z1, H0, Om)
  END DO

RETURN
END SUBROUTINE Initial_guess

SUBROUTINE Displacement (No_data,DM_smooth_data, DM_data,sigma_data,shift_best,chi_smooth,chi_min)
	IMPLICIT NONE
	INTEGER,INTENT(IN) :: No_data
	REAL*8,INTENT(IN),DIMENSION(No_data) :: DM_smooth_data, DM_data,sigma_data
	REAL*8,INTENT(OUT) :: shift_best,chi_min
	real*8,intent(out),dimension(No_data) :: chi_smooth
	real*8,dimension(No_data) :: chi
	REAL*8 :: shift
	INTEGER :: i

	chi_min = 10D5
	shift_best = 0.0D0
	chi = 0.0D0

!   shift = 0.0D0
	shift = -0.05D0	! this is kinda marginalization over hubble, since hubble constant move data vertically
	DO WHILE (shift .LE. 0.05D0 )	

	DO i = 1, No_data
		chi(i) = (( (DM_smooth_data(i) + shift) - (DM_data(i))  ) / (sigma_data(i)) )
	END DO
	IF (sum(chi**2) .LE. chi_min) THEN
		shift_best = shift
		chi_min = sum(chi**2)
		chi_smooth = chi
!		PRINT*,"Guess model is shifted",shift_best,chi_min !To see the effect of data movement in each step
	END IF
!		PRINT*,"Guess model is shifted",shift,chi !To see the effect of data movement in each step
	shift = shift + 0.1D-3
	END DO
!		PRINT*,"Guess model is shifted",shift_best,chi_min !To see the effect of data movement in each itera
RETURN
END SUBROUTINE Displacement
!==========================================================================
!		Smoothing
!
!	This routine does the smoothing
!
!
!
!==========================================================================
SUBROUTINE Smoothing (No_data,No_bins,z_data,sigma_data,DM_data,DM_guess_data,DM_smooth_data,z,&
											DM_guess_bins,DM_smooth_bins,delta_ar_bin,delta_ar_data)
 IMPLICIT NONE


  REAL*8 :: kernel1,kernel
  INTEGER :: i,j
  INTEGER,INTENT(IN) :: No_data,No_bins
  REAL*8,INTENT(IN),DIMENSION(No_data) :: z_data,sigma_data,DM_data,delta_ar_data
  REAL*8,INTENT(INOUT),DIMENSION(No_data) :: DM_guess_data
  REAL*8,INTENT(OUT),DIMENSION(No_data) :: DM_smooth_data
  REAL*8,INTENT(IN),DIMENSION(No_bins) :: z,delta_ar_bin
  REAL*8,INTENT(INOUT),DIMENSION(No_bins) :: DM_guess_bins
  REAL*8,INTENT(OUT),DIMENSION(No_bins) ::	DM_smooth_bins

  DM_smooth_data = 0.0D0
  DM_smooth_bins = 0.0D0



  DO i = 1, No_data
     CALL Normalization2 (z_data,delta_ar_data,No_data, i, sigma_data, DM_data, DM_guess_data, kernel1)
     DM_smooth_data (i) = DM_guess_data (i) + kernel1
 END DO

  DO i = 1, No_data
     DM_guess_data (i) = DM_smooth_data (i)
  END DO



  DO j = 1 , No_bins
	CALL Normalization (z_data,delta_ar_bin,z, No_bins, No_data, j, sigma_data, DM_data, DM_guess_data, kernel)
	DM_smooth_bins(j) = DM_guess_bins (j) + kernel
  END DO

  DO j = 1 , No_bins
    DM_guess_bins (j) = DM_smooth_bins (j)
  END DO


RETURN
END SUBROUTINE Smoothing
!================================================
!	This subroutine computes the kernel in smoothing formula for data points
!	INPUTS:
!	1) No_data1 : # of SN-data
!	2) z_data1: redshift of SN-data
!	3) DM_guess_data1: guess of distance modulus in data points
!	4) DM_data1: distance modulus of SN-data
!	5) sigma_data1: variance in distance modulus of SNs
!	6) Delta1: the smoothing width (Remind: it is asked in the program from the user
!
!
!	OUTPUT:
!	kernel2: it is the kernel term in smoothing formula
!================================================

! This routine calculte the kernel term for data points

SUBROUTINE Normalization2 (z_data1,delta_ar_data,No_data1, i1, sigma_data1, DM_data1, DM_guess_data1, kernel2)
  IMPLICIT NONE

  INTEGER :: m			
  INTEGER,INTENT (IN) :: No_data1, i1
  REAL*8,DIMENSION(No_data1),INTENT(IN) :: z_data1, DM_data1, sigma_data1,delta_ar_data
  REAL*8,DIMENSION(No_data1),INTENT(IN) :: DM_guess_data1
  REAL*8,INTENT(OUT) :: kernel2
  REAL*8 :: N,z

  kernel2 = 0.0D0
  N = 0.0D0
  DO m = 1, No_data1
    N = N + ( (EXP(-0.5D0*(((log((1.0D0+z_data1(m))/(1.0D0+z_data1(i1))))/(delta_ar_data(i1)))**2)))*(1.0D0/((sigma_data1(m))**2)))
  END DO




  DO m = 1, No_data1
    kernel2 = kernel2 + ( ((DM_data1(m)) - (DM_guess_data1(m))) / ((sigma_data1(m))**2) ) * &
     (EXP(-0.5D0*(((log((1.0D0+z_data1(m))/(1.0D0+z_data1(i1))))/(delta_ar_data(i1)))**2)))
  END DO
  kernel2 = (1.0D0/(N))*(kernel2)

RETURN
END SUBROUTINE Normalization2


!	 This routine calculte the kernel term for grids

SUBROUTINE Normalization (z_data1,delta_ar_bin, z1, No_bins1, No_data1, j1, sigma_data1, DM_data1, DM_guess_data1, kernel2)
  IMPLICIT NONE

  INTEGER :: m			
  INTEGER,INTENT (IN) :: No_data1, j1, No_bins1

  REAL*8,DIMENSION(No_data1),INTENT(IN) :: z_data1, DM_data1, sigma_data1
  REAL*8,DIMENSION(No_data1),INTENT(IN) :: DM_guess_data1
  REAL*8,DIMENSION(No_bins1),INTENT(IN) :: z1,delta_ar_bin
  REAL*8,INTENT(OUT) :: kernel2
  REAL*8 :: N,z


  kernel2 = 0.0D0
  N = 0.0D0
  DO m = 1, No_data1
    N = N + ((EXP(-0.5D0*(((log((1.0D0+z_data1(m))/(1.0D0+z1(j1))))/(delta_ar_bin(j1)))**2)))* &
   (1.0D0/((sigma_data1(m))**2)))
  END DO



  DO m = 1, No_data1
    kernel2 = kernel2 + ( ((DM_data1(m)) - (DM_guess_data1(m))) / ((sigma_data1(m))**2) ) * &
    (EXP(-0.5D0*(((log((1.0D0+z_data1(m))/(1.0D0+z1(j1))))/(delta_ar_bin(j1)))**2)))
  END DO
  kernel2 = (1.0D0/(N))*(kernel2)


RETURN
END SUBROUTINE Normalization

END MODULE Routines
