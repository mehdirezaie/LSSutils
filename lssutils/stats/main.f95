!==========================================================================
!  Program file name: main.f95                                            !
!                                                                         !
!  © Mehdi Rezaie                                                         !
!                                                                         !
!  Last modified: 2014-10-17 14:27:50                                     !
!                                                                         !
! This program calcualte chi-square between data and flat LCDM model	  !
!                                                                         !
!                                                                         !
!==========================================================================
PROGRAM Main

  USE Routines
  IMPLICIT NONE


  character(len = 20) :: filename,gnu = "gnuplot plot.gnu"
  integer :: i,f,t,l,j,n = 0,n_iteration = 5000 !n # of data
  integer :: n_bins = 300
  real*8 :: p = 3.0D0,del = 1.4D0,alpha,y,norm_factor = 1.96D0

  real*8,allocatable :: z_data(:),dm_data(:),sigma_data(:),chi_temp(:),chi_true(:),chi_lc(:),DM_guess_bins(:),DM_guess_data(:), &
	delta_ar_bin (:), delta_ar_data (:),z(:),DM_smooth_data(:),DM_smooth_bins(:),chi_smooth(:),H(:),q(:)




  real*8 :: x,H0,Om !km/s.Mpc
  real*8 :: chi_min,chi_marginalized,binsize,shift_best


! data reading


!			Argument reading: filename is the name of input which includes the data
!========================================================================

  CALL get_command_argument(1, filename)
  IF (LEN_TRIM(filename) == 0) WRITE(*,*) "Error"
  WRITE(filename,*) TRIM(filename)





!			Opening files
!========================================================================

  OPEN(1, FILE = "inputs/"//TRIM(adjustl(filename)),STATUS = "OLD",ACTION="READ")
  open(2, file = "outputs/chi/chi_"//TRIM(adjustl(filename)),position = "rewind", action = "write")
  open(3, file = "outputs/chi/total_chi_"//TRIM(adjustl(filename)))
  open(10,file = "outputs/mu/mu_"//TRIM(adjustl(filename)))
  open(11,file = "outputs/h/h_"//TRIM(adjustl(filename)))
  open(12,file = "outputs/q/q_"//TRIM(adjustl(filename)))



!			# of data is specified
!
  do 
	read(1,*,end = 10)
    n = n + 1
  end do
10 rewind(1)

  print*,n


!		Allocating arrays
!


  allocate(z_data(n),dm_data(n),sigma_data(n),chi_temp(n),chi_true(n),chi_lc(n),DM_guess_bins(n_bins),DM_guess_data(n), &
	delta_ar_bin(n_bins), delta_ar_data(n),z(n_bins),DM_smooth_data(n),DM_smooth_bins(n_bins),chi_smooth(n),&
	H(n_bins),q(n_bins))

  do i =1,n
	read(1,*)z_data(i),dm_data(i),sigma_data(i)
  end do








! generating true model to calculate its chi-square
!========================================================================

  chi_temp = 0.0D0
  chi_true = 0.0D0
  H0 = 68.0D0 ! km/s.Mpc
  chi_min = 50.0D3
  do j = 1,50
    do i = 1, n
    	x = z_data(i)
    	chi_temp(i) = ((Kink_mu (x, H0)-dm_data(i))/(sigma_data(i)))
    end do

    if (sum(chi_temp**2) .LT. chi_min) then
		chi_true = chi_temp
		chi_min = sum(chi_temp**2)
		!print*,"chi min achieved once!",H0
	end if
  H0 = H0 + 0.1
  end do



!	file to write the chi-square of true model

  write(2,'("# z - chi_true")')
  do i =1,n	
    write(2,'(2(e26.15))')z_data(i),chi_true(i)
  end do



!  finding chi-square of best LCDM model
  chi_temp = 0.0D0

  chi_min = 5.0D3
  H0 = 68.0D0
  DO l = 1, 50
	 Om = 0.1D0

	 DO f = 1,30
	     DO i = 1, n
	       x = z_data(i)
	       chi_temp(i) =  ((  DM_flat (x,H0,Om)) - (dm_data(i))  ) / (sigma_data(i)) 
	     END DO
 	   if (sum(chi_temp**2) .LT. chi_min) then
			chi_lc = chi_temp
			chi_min = sum((chi_temp)**2)
!			print*,"chi min achieved once!",H0
		end if

	 Om = Om + 0.01D0
	 END DO
  H0 = H0 + 0.1D0
  END DO	
  write(2,'(2/,"# z - chi_LCDM")')
  do i =1,n	
    write(2,'(2(e26.15))')z_data(i),chi_lc(i)
  end do




! writing total values of chi-square


  write(3,'("# index 0 - true",/,"#index 1 - LCDM",/,"# index 2:5001 - reconstructed result ")')
  write(3,'("#",e26.15)')sum(chi_true**2)
  write(3,'("#",e26.15)')sum(chi_lc**2)




! smoothing

! binsize
! given no of bins, the binsize is evaluated for the smoothing procedure

  binsize = (MAXVAL(z_data))/Float(n_bins)

  IF (2.0*binsize .GT. MINVAL(z_data)) STOP "*** # of bins is low ***"





!	Calculate alpha, as a normalization factor of Delta(z) and specify the grids given # of bins
!========================================================================

! Determining the redshifts over the grids and normalizing alpha

	alpha = 1.0D0
	y = 0.0D0
	   x = MINVAL(z_data)-(1.5D0*binsize)
	   do i = 1, n_bins
	   z(i) = x
	   y = y + delta2(n,x,p,del,alpha,z_data,sigma_data)
       x = x + binsize
       end do
    alpha = (norm_factor)*(1.0D0/(binsize*y))	! alpha is normalized to 1.0



! Determining Delta on the bins and data points, for later use (smoothing procedure)!

	  do i = 1, n_bins
		x=z(i)
	    delta_ar_bin(i) = delta2(n,x,p,del,alpha,z_data,sigma_data)
     end do

      do i = 1, n
		x=z_data(i)
	    delta_ar_data(i) = delta2(n,x,p,del,alpha,z_data,sigma_data)
      end do



!	initial guess is produced by Om = 0.3 and H0 = 70.0

  H0 = 70.0D0
  Om = 0.30D0
  call Initial_guess (n,n_bins,z_data,H0,binsize,Om,DM_guess_data,DM_guess_bins)
  call Displacement (n,DM_guess_data,dm_data,sigma_data,shift_best,chi_smooth,chi_marginalized)

  DM_guess_data = DM_guess_data + shift_best
  DM_guess_bins = DM_guess_bins + shift_best





! smoothing is started
!========================================================================
  write(2,'(2/,"# z - chi_reconstructed")')
  write(10,'("# first raw is redshift - the rests are d modulus",/,300e26.15)')z

  do t = 1,n_iteration

    write(*,*)"j is started",t

    call Reconstruction (n_bins,DM_guess_bins,z,H,q)
    write(10,'(300e26.15)')DM_guess_bins
    write(11,'(300e26.15)')H
    write(12,'(300e26.15)')q

	call Smoothing (n,n_bins,z_data,sigma_data,dm_data,DM_guess_data,DM_smooth_data,z,&
						DM_guess_bins,DM_smooth_bins,delta_ar_bin,delta_ar_data)

	call Displacement (n,DM_guess_data,dm_data,sigma_data,shift_best,chi_smooth,chi_marginalized)
	DM_guess_data = DM_guess_data + shift_best
	DM_guess_bins = DM_guess_bins + shift_best
 


 	if (t .le. n_iteration .and. MOD(t,10) .eq. 0)then
  		 do i =1,n	
 		   write(2,'(2(e26.15))')z_data(i),chi_smooth(i)
 		 end do
 		 write(2,'(/)')
 		 write(3,'(I4,1x,e26.15)')t,sum(chi_smooth**2)
    end if

  end do




deallocate (z_data,dm_data,sigma_data,chi_temp,chi_true,chi_lc,DM_guess_bins,DM_guess_data,delta_ar_bin,delta_ar_data,z,&
	DM_smooth_data,DM_smooth_bins,chi_smooth,H,q)
END PROGRAM Main
