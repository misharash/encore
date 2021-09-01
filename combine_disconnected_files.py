### combine_disconnected_files.py (Oliver Philcox, 2021)
# This reads in a set of (data-random) and (random) particle counts and uses them to construct the disconnected N-point functions, including edge-correction
# It is designed to be used with the run_npcf.csh script
# Currently only the 4PCF is supported. The code will try to load the edge correction coupling matrices from the coupling_matrices/ directory, and recompute them if not
# The output is saved to the working directory with the same format as the NPCF counts, with the filename ...zeta_discon_{N}pcf.txt
# Note that we optionally save the output RRR coupling matrix as a .npy file, which is of use if many simulations with the same geometry are being analyzed.
# This is triggered if one enters a file name for the outRcoupling input. The file will be created if not present, and loaded otherwise.

import sys, os, time
import subprocess
import numpy as np
import multiprocessing
from sympy.physics.wigner import wigner_3j

## First read-in the input file string from the command line
if len(sys.argv)!=3 and len(sys.argv)!=4:
    raise Exception("Need to specify the input files and N!")
else:
    inputs = str(sys.argv[1])
    N = int(sys.argv[2])
if len(sys.argv)==4:
    outRcoupling = str(sys.argv[3])
else:
    outRcoupling = ''

print("Reading in files starting with %s\n"%inputs)
init = time.time()

if N!=4:
    raise Exception("Only N=4 is implemented!")

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

#################### COMPUTE XI_LM ####################

#### Load in RR_lm piece
R_file = inputs+'.r_2pcf_mult1.txt'
countsR = np.loadtxt(R_file,skiprows=7) # skipping rows with radial bins
l1_lm, m1_lm = np.asarray(countsR[:,:2],dtype=int).T
bin1_lm = np.asarray(np.loadtxt(R_file,skiprows=6,max_rows=1),dtype=int)
RR_lm = countsR[:,2::2]+1.0j*countsR[:,3::2]

n_ell = len(np.unique(l1_lm))
lmax = n_ell-1
n_r = len(bin1_lm)
nlm = n_ell**2

#### Load in NN_lm piece
countsN_all = []
total_DmR = 0
for i in range(100):
    DmR_file = inputs+'.n%s_2pcf_mult1.txt'%(str(i).zfill(2))
    if not os.path.exists(DmR_file): continue
    # Extract counts
    tmp_counts = np.loadtxt(DmR_file,skiprows=7)
    countsN_all.append(tmp_counts[:,2::2]+1.0j*tmp_counts[:,3::2])
countsN_all = np.asarray(countsN_all)
N_files = len(countsN_all)
NN_lm = np.mean(countsN_all,axis=0)

#### Compute coupling matrix if necessary
coupling_file = get_script_path()+'/coupling_matrices/disconnected_coupling_matrix1_lmax%d.npy'%lmax
try:
    C_matrix = np.load(coupling_file)
    print("Loading coupling matrix from file")
except IOError:
    print("Computing xi_lm coupling matrix for l_max = %d"%lmax)

    # Form Gaunt matrix
    C_matrix = np.zeros((nlm,nlm,nlm))

    for l in range(n_ell):
        pref1 = np.sqrt((2.*l+1.)/(4.*np.pi))
        for lp in range(n_ell):
            pref2 = pref1*np.sqrt((2.*lp+1.))
            for L in range(n_ell):
                pref3 = pref2*np.sqrt((2.*L+1.))
                tj1 = wigner_3j(l,lp,L,0,0,0)
                if tj1==0: continue
                for m in range(-l,l+1):
                    for mp in range(-lp,lp+1):
                        M = m+mp
                        if np.abs(M)>L: continue
                        tj2 = tj1*wigner_3j(l,lp,L,m,mp,-M)
                        if tj2==0: continue
                        C_matrix[l**2+l+m,lp**2+lp+mp,L**2+L+M] = pref3*tj2*(-1.)**M

    # Save matrix to file
    np.save(coupling_file,C_matrix)
    print("\nSaved coupling matrix to %s"%coupling_file)

#### Edge-correct xi_lm

# First define f_array as RR_lm / RR_00
f_array = RR_lm[:,:]/RR_lm[0,:]

# Form relevant coupling matrix
coupling_matrix = np.zeros((nlm,nlm,n_r))+0.j

for n in range(nlm):
    for Ni in range(nlm):
        tmp_sum = 0.
        for n_p in range(nlm):
            tmp_sum += C_matrix[n,n_p,Ni]*f_array[n_p]
        coupling_matrix[n,Ni,:] = tmp_sum

xi_lm = np.zeros_like(NN_lm)
for i in range(len(NN_lm[0])):
    xi_lm[:,i] = np.matmul(np.linalg.inv(coupling_matrix[:,:,i]),NN_lm[:,i]/RR_lm[0,i])

print("Computed edge-corrected xi_lm multipoles after %.2f seconds"%(time.time()-init))

#################### COMPUTE XI_{LML'M'} ####################

#### Load in RRR_{lml'm'} piece
R_file = inputs+'.r_2pcf_mult2.txt'
countsR = np.loadtxt(R_file,skiprows=8) # skipping rows with radial bins
l1_lmlm, m1_lmlm, l2_lmlm, m2_lmlm = np.asarray(countsR[:,:4],dtype=int).T
bin1_lmlm, bin2_lmlm = np.asarray(np.loadtxt(R_file,skiprows=6,max_rows=2),dtype=int)
RRR_lmlm = countsR[:,4::2]+1.0j*countsR[:,5::2]

assert len(np.unique(l1_lmlm))==n_ell
assert len(np.unique(bin1_lm))==n_r

#### Load in RNN_{lml'm'} piece
countsN_all = []
total_DmR = 0
for i in range(100):
    DmR_file = inputs+'.n%s_2pcf_mult2.txt'%(str(i).zfill(2))
    if not os.path.exists(DmR_file): continue
    # Extract counts
    tmp_counts = np.loadtxt(DmR_file,skiprows=8)
    countsN_all.append(tmp_counts[:,4::2]+1.0j*tmp_counts[:,5::2])
countsN_all = np.asarray(countsN_all)
assert len(countsN_all)==N_files
RNN_lmlm = np.mean(countsN_all,axis=0)*-1. # add -1 due to weight inversion

if len(outRcoupling)>0 and os.path.exists(outRcoupling):
    print("Loading RRR coupling matrix from file!")
    coupling_matrix2 = np.load(outRcoupling)

else:
    # Compute collapsed indices
    index1 = l1_lmlm**2+l1_lmlm+m1_lmlm
    index2 = l2_lmlm**2+l2_lmlm+m2_lmlm

    # Define f2_array (from RRR_{lml'm'} / RRR_0000)
    f2_array = RRR_lmlm[:,:]/RRR_lmlm[0,:]

    # Form coupling matrix
    coupling_matrix2 = np.zeros((nlm**2,nlm**2,len(RRR_lmlm[0])))+0.j

    C_mat1 = C_matrix[index1][:,:,index1] # compute C_{l1,:,L1}^{m1,:,M1} for all n1, N1 arrays
    C_mat2 = C_matrix[index2][:,:,index2] # compute C_{l2,:,L2}^{m2,:,M2} for all n2, N2 arrays

    print("Computing xi_{lml'm'} edge-correction matrix")
    for prime_index in range(nlm**2):
        if prime_index%10==0: print("Accumulating primed-index %d of %d"%(prime_index+1,nlm**2))
        this_C1 = C_mat1[:,index1[prime_index],:]
        this_C2 = C_mat2[:,index2[prime_index],:]
        coupling_matrix2[:,:,:] += (this_C1*this_C2)[:,:,np.newaxis]*f2_array[prime_index]

    if len(outRcoupling)>0:
        print("Saving RRR coupling matrix to file")
        np.save(outRcoupling,coupling_matrix2)
    
print("Performing edge-correction")
xi_lmlm = np.zeros_like(RNN_lmlm)
for i in range(len(RNN_lmlm[0])):
    xi_lmlm[:,i] = np.matmul(np.linalg.inv(coupling_matrix2[:,:,i]),RNN_lmlm[:,i]/RRR_lmlm[0,i])

print("Computed edge-corrected xi_{lml'm'} multipoles after %.2f seconds"%(time.time()-init))

#################### COMBINE TO OBTAIN 4PCF ####################

#### Compute 4PCF coupling matrix if necessary
coupling_file = get_script_path()+'/coupling_matrices/disconnected_4pcf_coupling_lmax%d.npy'%lmax
try:
    fourpcf_coupling = np.load(coupling_file)
    print("Loading 4PCF coupling matrix from file")
except IOError:
    print("Computing 4PCF coupling matrix for l_max = %d"%lmax)

    # Form 4PCF coupling matrix, i.e. ThreeJ[l1,l2,l3,m1,m2,m3]
    fourpcf_coupling = np.zeros((nlm,nlm,nlm))
    for l in range(n_ell):
        for m in range(-l,l+1):
            for lp in range(n_ell):
                for mp in range(-lp,lp+1):
                    for L in range(n_ell):
                        for M in range(-L,L+1):
                            tj = wigner_3j(l,lp,L,m,mp,M)
                            if tj==0: continue
                            fourpcf_coupling[l**2+l+m,lp**2+lp+mp,L**2+L+M] = tj

    # Save matrix to file
    np.save(coupling_file,fourpcf_coupling)
    print("\nSaved 4PCF coupling matrix to %s"%coupling_file)

### Define an output matrix shape and arrays
ct_ell = 0
ell_1, ell_2, ell_3 = [],[],[]
for l1 in range(lmax):
    for l2 in range(lmax):
        for l3 in range(lmax):
            if pow(-1.,l1+l2+l3)==-1: continue
            if l3<np.abs(l1-l2): continue
            if l3>l1+l2: continue
            ct_ell+=1
            ell_1.append(l1)
            ell_2.append(l2)
            ell_3.append(l3)

ct_r = 0
bin1, bin2, bin3 = [],[],[]
for b1 in range(n_r):
    for b2 in range(b1+1,n_r):
        for b3 in range(b2+1,n_r):
            ct_r += 1
            bin1.append(b1)
            bin2.append(b2)
            bin3.append(b3)

zeta_discon = np.zeros((ct_ell,ct_r))

# Sum to accumulate 4PCF
print("Accumulating disconnected %dPCF"%N)
for r_index in range(ct_r):
    b1,b2,b3 = bin1[r_index],bin2[r_index],bin3[r_index]

    ## First permutation

    # Find relevant radial indices
    xi1 = xi_lm[:,bin1_lm==b1][:,0]
    xi2 = xi_lmlm[:,np.logical_and(bin1_lmlm==b2,bin2_lmlm==b3)][:,0]

    # Find angular bins and add to sum
    for l_index in range(ct_ell):
        ell1,ell2,ell3 = ell_1[l_index],ell_2[l_index],ell_3[l_index]

        this_xi1 = xi1[l1_lm==ell1]
        this_xi2 = xi2[np.logical_and(l1_lmlm==ell2,l2_lmlm==ell3)]
        this_coupling = fourpcf_coupling[l1_lm==ell1][:,index1[np.logical_and(l1_lmlm==ell2,l2_lmlm==ell3)],index2[np.logical_and(l1_lmlm==ell2,l2_lmlm==ell3)]]

        zeta_discon[l_index,r_index] += np.real_if_close(np.sum(this_xi1[:,np.newaxis]*this_xi2[np.newaxis,:]*this_coupling))

    ## Second permutation

    # Find angular bins and add to sum
    xi1 = xi_lm[:,bin1_lm==b2][:,0]
    xi2 = xi_lmlm[:,np.logical_and(bin1_lmlm==b1,bin2_lmlm==b3)][:,0]

    for l_index in range(ct_ell):
        ell1,ell2,ell3 = ell_1[l_index],ell_2[l_index],ell_3[l_index]

        this_xi1 = xi1[l1_lm==ell2]
        this_xi2 = xi2[np.logical_and(l1_lmlm==ell1,l2_lmlm==ell3)]
        this_coupling = fourpcf_coupling[l1_lm==ell2][:,index1[np.logical_and(l1_lmlm==ell1,l2_lmlm==ell3)],index2[np.logical_and(l1_lmlm==ell1,l2_lmlm==ell3)]]

        zeta_discon[l_index,r_index] += np.real_if_close(np.sum(this_xi1[:,np.newaxis]*this_xi2[np.newaxis,:]*this_coupling))

    ## Third permutation

    # Find angular bins and add to sum
    xi1 = xi_lm[:,bin1_lm==b3][:,0]
    xi2 = xi_lmlm[:,np.logical_and(bin1_lmlm==b1,bin2_lmlm==b2)][:,0]

    for l_index in range(ct_ell):
        ell1,ell2,ell3 = ell_1[l_index],ell_2[l_index],ell_3[l_index]

        this_xi1 = xi1[l1_lm==ell3]
        this_xi2 = xi2[np.logical_and(l1_lmlm==ell1,l2_lmlm==ell2)]
        this_coupling = fourpcf_coupling[l1_lm==ell3][:,index1[np.logical_and(l1_lmlm==ell1,l2_lmlm==ell2)],index2[np.logical_and(l1_lmlm==ell1,l2_lmlm==ell2)]]

        zeta_discon[l_index,r_index] += np.real_if_close(np.sum(this_xi1[:,np.newaxis]*this_xi2[np.newaxis,:]*this_coupling))

# Now save the output to file, copying the first few lines from the N files
zeta_file = inputs+'.zeta_discon_%dpcf.txt'%N
R_file = inputs+'.r_3pcf.txt'
rfile = open(R_file,"r")
zfile = open(zeta_file,"w")
for l,line in enumerate(rfile):
    if l>=4: continue
    zfile.write(line)
zfile.write("## Format: Row 1 = radial bin 1, Row 2 = radial bin 2, Row 3 = radial bin 3, Rows 4+ = zeta^{(disc)}_l1l2l3^abc\n");
zfile.write("## Columns 1-3 specify the (l1, l2, l3) multipole triplet\n");
zfile.write("\t\t\t")
for i in range(ct_r): zfile.write("%d\t"%bin1[i])
zfile.write("\n")
zfile.write("\t\t\t")
for i in range(ct_r): zfile.write("%d\t"%bin2[i])
zfile.write("\n")
zfile.write("\t\t\t")
for i in range(ct_r): zfile.write("%d\t"%bin3[i])
zfile.write("\n")

for a in range(ct_ell):
    zfile.write("%d\t"%ell_1[a])
    zfile.write("%d\t"%ell_2[a])
    zfile.write("%d\t"%ell_3[a])
    for b in range(ct_r):
        zfile.write("%.8e\t"%zeta_discon[a,b])
    zfile.write("\n")
zfile.close()

print("Disconnected %dPCF saved to %s"%(N,zeta_file))
print("Exiting after %.2f seconds"%(time.time()-init))
sys.exit();
