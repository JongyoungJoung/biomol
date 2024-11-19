!
! extract binding site interface residues
! input  : atom coordinate list, residue name list
!          protein 1 = receptor / protein 2 = ligand
! output : pairs list
!
subroutine get_interface_residues(crd1,ncrd1,crd2,ncrd2,cutoff,pairs,num_pairs)
implicit none
! crd list : (x,y,z,res_id)
integer,intent(in) :: ncrd1,ncrd2
real*8,intent(in)  :: cutoff
real*8,intent(in)  :: crd1(ncrd1,4)  ! atomic coordinate list of receptor
real*8,intent(in)  :: crd2(ncrd2,4)  ! atomic coordinate list of ligand
integer,intent(out) :: pairs(ncrd1*ncrd2,2)    ! output, residue-residue pairs
integer,intent(out) :: num_pairs

integer :: i,j
integer :: ii
real*8 :: cutoff2
real*8 :: dist
logical :: check
integer :: elem1, elem2
!integer :: temp_id_pair(ncrd1*ncrd2,2)

cutoff2=cutoff*cutoff
dist = 0.0d0
num_pairs=1
!temp_id_pairs = 0
check = .false.

do i=1,ncrd1
    elem1 = int(crd1(i,4))
    do j=1,ncrd2
    call caldistance2(crd1(i,1:3), crd2(j,1:3), dist)
    if (dist.le.cutoff2) then
        elem2 = int(crd2(j,4))
        check = .false.
        do ii=1,num_pairs   ! check this residue pair already exist
            if ((pairs(ii,1).eq.elem1).and.(pairs(ii,2).eq.elem2)) then
                check = .true.
            endif
        enddo
        if (check .neqv. .true.) then 
            pairs(num_pairs,1) = int(crd1(i,4)) ! receptor's residue
            pairs(num_pairs,2) = int(crd2(j,4)) ! ligand's residue
            num_pairs = num_pairs + 1
        endif
    endif
    enddo
enddo

num_pairs = num_pairs - 1

end subroutine get_interface_residues
!!!!!!!!!!!!!!
subroutine get_intra_residue_pairs(crd,ncrd,cutoff,pairs,num_pairs)
implicit none
integer,intent(in) :: ncrd
real*8,intent(in)  :: cutoff
real*8,intent(in)  :: crd(ncrd,4)
integer,intent(out) :: pairs(ncrd*ncrd,2)
integer,intent(out) :: num_pairs

integer :: i,j,ii
real*8 :: cutoff2,dist
logical :: check
integer :: elem1,elem2

pairs = -1
num_pairs=0

cutoff2=cutoff*cutoff
dist=0.0d0
do i=1,ncrd-1
    elem1 = int(crd(i,4))
    do j=i+1,ncrd
        call caldistance2(crd(i,1:3),crd(j,1:3),dist)
        !print *,i,j,dist
        if (dist.le.cutoff2) then
            elem2 = int(crd(j,4))
            check = .false.
            do ii=1,num_pairs
                if ((pairs(ii,1).eq.elem1).and.(pairs(ii,2).eq.elem2)) then
                    check = .true.
                endif
            enddo
            if (check .neqv. .true.) then
                pairs(num_pairs,1) = int(crd(i,4))
                pairs(num_pairs,2) = int(crd(j,4))
                num_pairs = num_pairs + 1
            endif
        endif
    enddo
enddo
end subroutine get_intra_residue_pairs
!!!!!!!!!!!!!!
subroutine get_rmsd_native_vs_model(natpdb,modpdb,rmsd,nnat,nmod)
!   nnat_rec == nmod_rec
!   nnat_lig == nmod_lig
implicit none
integer,intent(in) :: nnat,nmod
real*8,intent(in) :: natpdb(nnat,3)
real*8,intent(in) :: modpdb(nmod,3)

real*8,intent(out) :: rmsd

integer :: i
real*8 :: dx,dy,dz

rmsd = 0.0d0

do i=1,nnat
    dx = natpdb(i,1)-modpdb(i,1)
    dy = natpdb(i,2)-modpdb(i,2)
    dz = natpdb(i,3)-modpdb(i,3)
    rmsd = rmsd + (dx*dx + dy*dy + dz*dz)
    !print *,natpdb(i,:)
    !print *,modpdb(i,:)
    !print *,dx,dy,dz
    !print *,""
enddo
rmsd = sqrt(rmsd/float(nnat))

end subroutine get_rmsd_native_vs_model
!!!!!!!!!!!!!!
subroutine get_rmsd_between_2_protein(pdb1,pdb2,rmsd,n1,n2)
!   nnat_rec == nmod_rec
!   nnat_lig == nmod_lig
implicit none
integer,intent(in) :: n1,n2
real*8,intent(in) :: pdb1(n1,3)
real*8,intent(in) :: pdb2(n2,3)

real*8,intent(out) :: rmsd

integer :: i
real*8 :: dx,dy,dz

rmsd = 0.0d0

do i=1,n1
    dx = pdb1(i,1)-pdb2(i,1)
    dy = pdb1(i,2)-pdb2(i,2)
    dz = pdb1(i,3)-pdb2(i,3)
    rmsd = rmsd + (dx*dx + dy*dy + dz*dz)
    !print *,natpdb(i,:)
    !print *,modpdb(i,:)
    !print *,dx,dy,dz
    !print *,""
enddo
rmsd = sqrt(rmsd/float(n1))

end subroutine get_rmsd_between_2_protein
!!!!!!!!!!!!!!
subroutine caldistance2(vec1, vec2, dist)
implicit none
real*8, intent(in)  :: vec1(3)
real*8, intent(in)  :: vec2(3)
real*8, intent(out) :: dist
integer :: i

dist = 0.0

do i=1,3
    dist = dist + (vec1(i)-vec2(i))*(vec1(i)-vec2(i))
enddo
end subroutine calDistance2
!!!!!!!!!!!!!!
subroutine centroid(crd,ncrd,center)
implicit none
integer,intent(in) :: ncrd 
real*8,intent(in)  :: crd(ncrd,3)
real*8,intent(out) :: center(3)
integer :: i

center = 0.0d0
do i=1,ncrd
    center(1) = center(1) + crd(i,1)
    center(2) = center(2) + crd(i,2)
    center(3) = center(3) + crd(i,3)
enddo
center(1) = center(1)/float(ncrd)
center(2) = center(2)/float(ncrd)
center(3) = center(3)/float(ncrd)

end subroutine centroid
!!!!!!!!!!!!!!
subroutine translate(crd,ncrd,center)
implicit none
integer,intent(in)   :: ncrd
real*8,intent(inout) :: crd(ncrd,3)
real*8,intent(in)    :: center(3)
integer :: i

do i=1,ncrd
    crd(i,1) = crd(i,1) - center(1)
    crd(i,2) = crd(i,2) - center(2)
    crd(i,3) = crd(i,3) - center(3)
enddo

end subroutine translate
!!!!!!!!!!!!!!
!   subroutine superpose (n1,x1,y1,z1,n2,x2,y2,z2)
!   !
!   !     origin : quadfit in Tinker
!   !     modified for f2py
!   !
!   implicit none
!   !include 'sizes.i'
!   !include 'align.i'
!   integer i,i1,i2,n1,n2
!   integer,parameter :: maxatm = 20000
!   real*8 weigh,xrot,yrot,zrot
!   real*8 xxyx,xxyy,xxyz
!   real*8 xyyx,xyyy,xyyz
!   real*8 xzyx,xzyy,xzyz
!   real*8 q(4),d(4)
!   real*8 work1(4),work2(4)
!   real*8 rot(3,3)
!   real*8 c(4,4),v(4,4)
!   !real*8 x1(maxatm),x2(maxatm)
!   !real*8 y1(maxatm),y2(maxatm)
!   !real*8 z1(maxatm),z2(maxatm)
!   real*8,allocatable :: x1(:),x2(:)
!   real*8,allocatable :: y1(:),y2(:)
!   real*8,allocatable :: z1(:),z2(:)

!   allocate (x1(n1))
!   allocate (y1(n1))
!   allocate (z1(n1))
!   allocate (x2(n2))
!   allocate (y2(n2))
!   allocate (z2(n2))

!   !build the upper triangle of the quadratic form matrix

!   xxyx = 0.0d0
!   xxyy = 0.0d0
!   xxyz = 0.0d0
!   xyyx = 0.0d0
!   xyyy = 0.0d0
!   xyyz = 0.0d0
!   xzyx = 0.0d0
!   xzyy = 0.0d0
!   xzyz = 0.0d0
!   do i = 1, nfit
!      i1 = ifit(1,i)
!      i2 = ifit(2,i)
!      weigh = wfit(i)
!      xxyx = xxyx + weigh*x1(i1)*x2(i2)
!      xxyy = xxyy + weigh*y1(i1)*x2(i2)
!      xxyz = xxyz + weigh*z1(i1)*x2(i2)
!      xyyx = xyyx + weigh*x1(i1)*y2(i2)
!      xyyy = xyyy + weigh*y1(i1)*y2(i2)
!      xyyz = xyyz + weigh*z1(i1)*y2(i2)
!      xzyx = xzyx + weigh*x1(i1)*z2(i2)
!      xzyy = xzyy + weigh*y1(i1)*z2(i2)
!      xzyz = xzyz + weigh*z1(i1)*z2(i2)
!   end do
!   c(1,1) = xxyx + xyyy + xzyz
!   c(1,2) = xzyy - xyyz
!   c(2,2) = xxyx - xyyy - xzyz
!   c(1,3) = xxyz - xzyx
!   c(2,3) = xxyy + xyyx
!   c(3,3) = xyyy - xzyz - xxyx
!   c(1,4) = xyyx - xxyy
!   c(2,4) = xzyx + xxyz
!   c(3,4) = xyyz + xzyy
!   c(4,4) = xzyz - xxyx - xyyy

!   !diagonalize the quadratic form matrix

!   call jacobi (4,4,c,d,v,work1,work2)

!   !extract the desired quaternion

!   q(1) = v(1,4)
!   q(2) = v(2,4)
!   q(3) = v(3,4)
!   q(4) = v(4,4)

!   !assemble rotation matrix that superimposes the molecules

!   rot(1,1) = q(1)**2 + q(2)**2 - q(3)**2 - q(4)**2
!   rot(2,1) = 2.0d0 * (q(2) * q(3) - q(1) * q(4))
!   rot(3,1) = 2.0d0 * (q(2) * q(4) + q(1) * q(3))
!   rot(1,2) = 2.0d0 * (q(3) * q(2) + q(1) * q(4))
!   rot(2,2) = q(1)**2 - q(2)**2 + q(3)**2 - q(4)**2
!   rot(3,2) = 2.0d0 * (q(3) * q(4) - q(1) * q(2))
!   rot(1,3) = 2.0d0 * (q(4) * q(2) - q(1) * q(3))
!   rot(2,3) = 2.0d0 * (q(4) * q(3) + q(1) * q(2))
!   rot(3,3) = q(1)**2 - q(2)**2 - q(3)**2 + q(4)**2

!   !rotate second molecule to best fit with first molecule

!   do i = 1, n2
!      xrot = x2(i)*rot(1,1) + y2(i)*rot(1,2) + z2(i)*rot(1,3)
!      yrot = x2(i)*rot(2,1) + y2(i)*rot(2,2) + z2(i)*rot(2,3)
!      zrot = x2(i)*rot(3,1) + y2(i)*rot(3,2) + z2(i)*rot(3,3)
!      x2(i) = xrot
!      y2(i) = yrot
!      z2(i) = zrot
!   end do

!   deallocate(x1)
!   deallocate(y1)
!   deallocate(z1)
!   deallocate(x1)
!   deallocate(y1)
!   deallocate(z1)

!   return
!   end subroutine superpose
!   !!!!!!!!!!!!!!
!   subroutine jacobi (n,np,a,d,v,b,z)
!   implicit none
!   include 'iounit.i'
!   integer i,j,k
!   integer n,np,ip,iq
!   integer nrot,maxrot
!   real*8 sm,tresh,s,c,t
!   real*8 theta,tau,h,g,p
!   real*8 d(np),b(np),z(np)
!   real*8 a(np,np),v(np,np)

!   ! setup and initialization

!   maxrot = 100
!   nrot = 0
!   do ip = 1, n
!      do iq = 1, n
!         v(ip,iq) = 0.0d0
!      end do
!      v(ip,ip) = 1.0d0
!   end do
!   do ip = 1, n
!      b(ip) = a(ip,ip)
!      d(ip) = b(ip)
!      z(ip) = 0.0d0
!   end do

!   ! perform the jacobi rotations

!   do i = 1, maxrot
!      sm = 0.0d0
!      do ip = 1, n-1
!         do iq = ip+1, n
!            sm = sm + abs(a(ip,iq))
!         end do
!      end do
!      if (sm .eq. 0.0d0)  goto 10
!      if (i .lt. 4) then
!         tresh = 0.2d0*sm / n**2
!      else
!         tresh = 0.0d0
!      end if
!      do ip = 1, n-1
!         do iq = ip+1, n
!            g = 100.0d0 * abs(a(ip,iq))
!            if (i.gt.4 .and. abs(d(ip))+g.eq.abs(d(ip)) &
!                       .and. abs(d(iq))+g.eq.abs(d(iq))) then
!               a(ip,iq) = 0.0d0
!            else if (abs(a(ip,iq)) .gt. tresh) then
!               h = d(iq) - d(ip)
!               if (abs(h)+g .eq. abs(h)) then
!                  t = a(ip,iq) / h
!               else
!                  theta = 0.5d0*h / a(ip,iq)
!                  t = 1.0d0 / (abs(theta)+sqrt(1.0d0+theta**2))
!                  if (theta .lt. 0.0d0)  t = -t
!               end if
!               c = 1.0d0 / sqrt(1.0d0+t**2)
!               s = t * c
!               tau = s / (1.0d0+c)
!               h = t * a(ip,iq)
!               z(ip) = z(ip) - h
!               z(iq) = z(iq) + h
!               d(ip) = d(ip) - h
!               d(iq) = d(iq) + h
!               a(ip,iq) = 0.0d0
!               do j = 1, ip-1
!                  g = a(j,ip)
!                  h = a(j,iq)
!                  a(j,ip) = g - s*(h+g*tau)
!                  a(j,iq) = h + s*(g-h*tau)
!               end do
!               do j = ip+1, iq-1
!                  g = a(ip,j)
!                  h = a(j,iq)
!                  a(ip,j) = g - s*(h+g*tau)
!                  a(j,iq) = h + s*(g-h*tau)
!               end do
!               do j = iq+1, n
!                  g = a(ip,j)
!                  h = a(iq,j)
!                  a(ip,j) = g - s*(h+g*tau)
!                  a(iq,j) = h + s*(g-h*tau)
!               end do
!               do j = 1, n
!                  g = v(j,ip)
!                  h = v(j,iq)
!                  v(j,ip) = g - s*(h+g*tau)
!                  v(j,iq) = h + s*(g-h*tau)
!               end do
!               nrot = nrot + 1
!            end if
!         end do
!      end do
!      do ip = 1, n
!         b(ip) = b(ip) + z(ip)
!         d(ip) = b(ip)
!         z(ip) = 0.0d0
!      end do
!   end do

!   ! print warning if not converged

!   continue
!   if (nrot .eq. maxrot) then
!      write (*,*) ' JACOBI  --  Matrix Diagonalization not Converged'
!   end if

!   ! sort the eigenvalues and vectors

!   do i = 1, n-1
!      k = i
!      p = d(i)
!      do j = i+1, n
!         if (d(j) .lt. p) then
!            k = j
!            p = d(j)
!         end if
!      end do
!      if (k .ne. i) then
!         d(k) = d(i)
!         d(i) = p
!         do j = 1, n
!            p = v(j,i)
!            v(j,i) = v(j,k)
!            v(j,k) = p
!         end do
!      end if
!   end do
!   return
!   end
