#if 0
It’s sometimes necessary for a data region to
be in a different scope than the compute
region.
When this occurs, the present clause can be
used to tell the compiler data is already on
the device.
Since the declaration of A is now in a higher
scope, it’s necessary to shape A in the present
clause.
#endif

      module glob
      real, dimension(:), allocatable :: x
      contains
         subroutine sub( y )
         real, dimension(:) :: y
!$acc declare present(y,x)
!$acc kernels
         do i = 1, ubound(y,1)
            y(i) = y(i) + x(i)
         enddo
!$acc end kernels
         end subroutine
      end module

      subroutine roo( z )
      use glob
      real :: z(:)
!$acc data copy(z) copyin(x)
      call sub( z )
!$acc end data region
      end subroutine
