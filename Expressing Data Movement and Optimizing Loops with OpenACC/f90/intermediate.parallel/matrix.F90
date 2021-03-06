module matrix_mod
  implicit none
  type matrix
    sequence
    integer :: num_rows
    integer :: nnz
    integer, pointer :: row_offsets(:)
    integer, pointer :: cols(:)
    real(8), pointer :: coefs(:)
  end type matrix
  public :: allocate_3d_poission_matrix, free_matrix
  public :: matvec
  contains
  subroutine allocate_3d_poission_matrix(a, n)
    implicit none
    type(matrix) :: a
    integer      :: n, num_rows, nnz, offsets(27), &
                    zstride, ystride, idx,         &
                    i, j, x, y, z
    real(8)      :: coefs(27)
    integer, pointer :: arow_offsets(:),acols(:)
    real(8), pointer :: acoefs(:)

    num_rows = (n+1)*(n+1)*(n+1)
    nnz = 27*num_rows
    a%num_rows = num_rows
    allocate(a%row_offsets(num_rows+1))
    allocate(a%cols(nnz))
    allocate(a%coefs(nnz))
    
    arow_offsets => a%row_offsets
    acols => a%cols
    acoefs => a%coefs

    zstride = n*n
    ystride = n

    i=1
    do z=-1,1
      do y=-1,1
        do x=-1,1
          offsets(i) = zstride*z + ystride*y + x
          if((x.eq.0).and.(y.eq.0).and.(z.eq.0)) then
            coefs(i) = 27
          else
            coefs(i) = -1
          endif
          i = i + 1
        enddo
      enddo
    enddo

    idx = 1
    do i=1,num_rows
      arow_offsets(i) = idx
      do j=1,27
        n=i+offsets(j)
        if ((n.ge.1).and.(n.le.num_rows)) then
          a%cols(idx) = n
          a%coefs(idx) = coefs(j)
          idx = idx + 1
        endif
      enddo
    enddo

    arow_offsets(num_rows+1) = idx
    a%nnz = idx-1
    
    !$acc enter data copyin(arow_offsets,acols,acoefs)

    end subroutine allocate_3d_poission_matrix
  
  subroutine free_matrix(a)
    implicit none
    type(matrix) :: a
    integer, pointer :: arow_offsets(:),acols(:)
    real(8), pointer :: acoefs(:)
    
    arow_offsets => a%row_offsets
    acols => a%cols
    acoefs => a%coefs

    !$acc exit data delete(arow_offsets,acols,acoefs)
   
    deallocate(arow_offsets)
    deallocate(acols)
    deallocate(acoefs)
  end subroutine free_matrix

  subroutine matvec(a, x, y)
    implicit none
    type(matrix) :: a
    real(8)      :: x(:), y(:)
    integer      :: i, j, row_start, row_end, acol
    real(8)      :: tmpsum, acoef, xcoef
    integer, pointer :: arow_offsets(:),acols(:)
    real(8), pointer :: acoefs(:)
    
    arow_offsets => a%row_offsets
    acols => a%cols
    acoefs => a%coefs

    !$acc parallel loop present(arow_offsets,acols,acoefs)
    do i=1,a%num_rows
      tmpsum = 0.0d0
      row_start = arow_offsets(i)
      row_end   = arow_offsets(i+1)-1
      !$acc loop reduction(+:tmpsum)
      do j=row_start,row_end
        acol = acols(j)
        acoef = acoefs(j)
        xcoef = x(acol)
        tmpsum = tmpsum + acoef*xcoef
      enddo
      y(i) = tmpsum
    enddo

  end subroutine matvec

end module matrix_mod
