program cross
 implicit none


  character(len = 20) :: filename
  integer :: i,t,n_data = 580,n_sample = 500,n_bins = 300,n1,n2,n3
  real*8,allocatable :: chi_smooth(:,:),c(:),q(:,:),x_max(:,:),chi_square(:),sum_t(:),h(:),q_d(:),z(:),Dm(:)
  real*8 :: x = 0.0D0





!			Argument reading: filename is the name of input which includes the data
!========================================================================

  CALL get_command_argument(1, filename)
  IF (LEN_TRIM(filename) == 0) WRITE(*,*) "Error"
  WRITE(filename,*) TRIM(filename)

!	reading every 
 open(2, file = "outputs/chi/chi_"//TRIM(adjustl(filename)),position = "rewind", action = "read", status = "old")
 

 allocate(chi_smooth(n_sample,n_data))

 read(2,'(1166/)')
 do t =1,n_sample
  do i = 1,n_data	
    read(2,'(27x,e26.15)')chi_smooth(t,i)
  end do
    read(2,'(/)')
 end do
 close(2)



  allocate(c(n_sample),chi_square(n_sample))

! calculating T 0 
 c = sum(chi_smooth,dim = 2)


! calculating chi-square
 chi_square = sum(chi_smooth**2,dim = 2)




  allocate(q(n_sample,n_data))

!	calculating Q(n)
 q = 0.0D0
 do t =1,n_sample
    do n2 = 1,n_data
    q(t,n2) = sum(chi_smooth(t,1:n2))
    end do
 end do


  deallocate(chi_smooth)
  allocate(x_max(3,n_sample))

! writing T 1 "crossing"
! open(3,file = "t1.txt")
 do t =1,n_sample
  do n1 = 1,n_data-1

	x = (q(t,n1)**2)+((q(t,n_data)-q(t,n1))**2)
!    write(3,'(I4,E26.15)')n1,x

!	max of T1
		if (x .GT. x_max(1,t))x_max(1,t) = x

  end do 
! write(3,'(/)')
 end do




! writing T 2 "crossing
! open(4,file = "t2.txt")
 do t =1,n_sample
  do n1 = 1,n_data-2


	do n2 = n1+1,n_data-1

	x =(q(t,n1)**2)+((q(t,n2)-q(t,n1))**2)+((q(t,n_data)-q(t,n2))**2)
!    write(4,'(2I4,E26.15)')n1,n2,x

	if (x .GT. x_max(2,t))x_max(2,t) = x

	end do
  end do 
! write(4,'(/)')
 end do



! writing T 3 "crossing
! open(5,file = "t3.txt")
 do t =1,n_sample
 print*,t
  do n1 = 1,n_data-3

	do n2 = n1+1,n_data-2

	do n3 = n2+1,n_data-1

	 x = (q(t,n1)**2)+((q(t,n2)-q(t,n1))**2)+((q(t,n3)-q(t,n2))**2) + ((q(t,n_data)-q(t,n3))**2)
!    write(5,'(3I4,E26.15)')n1,n2,n3,x

! max of T 3
	if (x .GT. x_max(3,t))x_max(3,t) = x
	end do

	end do

  end do 
! write(5,'(/)')
 end do




  deallocate(q)

! writing iteration - T 0 - T1 max - T2 max - T3 max - chi-square
 open(5,file = "it_T_max_"//TRIM(adjustl(filename)))
 write(5,'("#iteration - T 0 - T1 max - T2 max - T3 max -  chi-square")')
 do t =1,n_sample
  write(5,'(I4,5e26.15)')t,c(t),x_max(1,t),x_max(2,t),x_max(3,t),chi_square(t)
 end do
 close(5)
  
 allocate(sum_t(n_sample))
 do t = 1,n_sample
   sum_t(t) = c(t)+x_max(1,t)+x_max(2,t)+x_max(3,t)
 end do

 deallocate(c,x_max,chi_square)

 

 open(10,file = "outputs/mu/mu_"//TRIM(adjustl(filename)),position = "rewind", action = "read", status = "old")
 open(11,file = "outputs/h/h_"//TRIM(adjustl(filename)),position = "rewind", action = "read", status = "old")
 open(12,file = "outputs/q/q_"//TRIM(adjustl(filename)),position = "rewind", action = "read", status = "old")

 allocate(z(n_bins),h(n_bins),q_d(n_bins),Dm(n_bins))
 read(10,'(/,300e26.15)')z
 read(11,*)
 read(12,*)


 t = minloc(sum_t,dim = 1)
 t = t - 1
 do i = 1,t
 read(10,*)
 read(11,*)
 read(12,*)
 end do
 

 read(10,'(300e26.15)')Dm
 read(11,'(300e26.15)')h
 read(12,'(300e26.15)')q_d
 close(10);close(11);close(12)


 
 open(13, file = "outputs/best_mhq/mhq_"//TRIM(adjustl(filename)),position = "rewind", action = "write", status = "replace")
 write(13,'("# min location of min of T0:T3 for sample",1x,A15,1x,"is",1x,I5)')filename,minloc(sum_t)
 write(13,'("# redshift - distance modulus - hubble - q")')
 do i = 3,n_bins-2
	write(13,'(4e26.15)')z(i),Dm(i),h(i),q_d(i)
 end do
 close(13)
 
 deallocate(z,h,q_d,Dm,sum_t)
end program cross
