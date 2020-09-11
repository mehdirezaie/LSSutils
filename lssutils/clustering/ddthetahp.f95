! ===================================================================
!   a Fortran routine to compute two point 
!   angular correlation function
!   (c) Mehdi Rezaie, Sep 25, 2019
!
!   create the signature file (it is provided though)
!   to overwrite, use `--overwrite-signature'
!   > f2py ddthetahp.f95 -m ddthetahp -h ddthetahp.pyf
!   build final module
!   > f2py -c ddthetahp.pyf ddthetahp.f95
!
!   see test.py for an example
!
! ===================================================================
subroutine ddthetahpauto(theta1, phi1, delta1, fpix1, bins, C, n1, m)
    ! auto correlation function 
    ! healpix based
    implicit none

    ! variables
    integer,intent(in)                  :: n1, m
    real(8),dimension(n1),intent(in)    :: theta1, phi1, delta1, fpix1
    real(8),dimension(m),intent(in)     :: bins
    real(8),dimension(2,m-1),intent(out) :: C

    ! local variables
    real(8) :: s, be, sinti, costi, sinpi, cospi, delta1i, fpix1i
    integer :: i,j,binid
    real(8), dimension(n1) :: cost1,sint1,cosp1,sinp1

    ! theta and phi
    cost1 = dcos(theta1)
    sint1 = dsin(theta1)
    cosp1 = dcos(phi1)
    sinp1 = dsin(phi1)


    C    = 0.0
    s    = 0.0

    do i = 1, n1
        ! temp array
        sinti   = sint1(i)
        costi   = cost1(i)
        sinpi   = sinp1(i)
        cospi   = cosp1(i)
        delta1i = delta1(i)
        fpix1i  = fpix1(i)

        do j = i+1, n1  ! no auto pairs
           s = sinti*sint1(j)*(cospi*cosp1(j) + sinpi*sinp1(j)) + (costi*cost1(j))
           be = bins(1) ! starts from the max sep angle
           binid = 0
           do while (s > be)
                binid = binid + 1
                be    = bins(binid+1)
           end do
           if ((binid .ge. 1) .and. (binid .le. m)) then
               C(1,binid) = C(1,binid) + delta1i*delta1(j)*fpix1i*fpix1(j)
               C(2,binid) = C(2,binid) + fpix1i*fpix1(j)
           end if
           s = 0
        end do
    end do

return
end subroutine ddthetahpauto

subroutine ddthetahpcross(theta1, phi1, delta1, fpix1, theta2, phi2, delta2, fpix2, bins, C, n1, n2, m)
    ! cross correlation function 
    ! healpix based
    implicit none

    ! variables
    integer,intent(in)                  :: n1, n2, m
    real(8),dimension(n1),intent(in)    :: theta1, phi1, delta1, fpix1
    real(8),dimension(n2),intent(in)    :: theta2, phi2, delta2, fpix2
    real(8),dimension(m),intent(in)     :: bins
    real(8),dimension(2,m-1),intent(out) :: C

    ! local variables
    real(8) :: s, be, sinti, costi, sinpi, cospi, delta1i, fpix1i
    integer :: i,j,binid
    real(8), dimension(n1) :: cost1,sint1,cosp1,sinp1
    real(8), dimension(n2) :: cost2,sint2,cosp2,sinp2

    ! theta and phi
    cost1 = dcos(theta1)
    sint1 = dsin(theta1)
    cosp1 = dcos(phi1)
    sinp1 = dsin(phi1)

    cost2 = dcos(theta2)
    sint2 = dsin(theta2)
    cosp2 = dcos(phi2)
    sinp2 = dsin(phi2)



    C    = 0.0
    s    = 0.0

    do i = 1, n1
        ! temp array
        sinti   = sint1(i)
        costi   = cost1(i)
        sinpi   = sinp1(i)
        cospi   = cosp1(i)
        delta1i = delta1(i)
        fpix1i  = fpix1(i)

        do j = 1, n2  ! no auto pairs
           s = sinti*sint2(j)*(cospi*cosp2(j) + sinpi*sinp2(j)) + (costi*cost2(j))
           be = bins(1) ! starts from the max sep angle
           binid = 0
           do while (s > be)
                binid = binid + 1
                be    = bins(binid+1)
           end do
           if ((binid .ge. 1) .and. (binid .le. m)) then
               C(1,binid) = C(1,binid) + delta1i*delta2(j)*fpix1i*fpix2(j)
               C(2,binid) = C(2,binid) + fpix1i*fpix2(j)
           end if
           s = 0
        end do
    end do

return
end subroutine ddthetahpcross
