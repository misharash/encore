### combine_files_new.py (Michael Rashkovetskyi, adapted from Oliver Philcox, 2022)
# This reads in a set of (data-random) and (random) particle counts and uses them to construct the N-point functions, including edge-correction
# It is designed to be used with the run_npcf.csh script
# Currently 2PCF, 3PCF, 4PCF, 5PCF and 6PCF are supported. The code will try to load the edge correction coupling matrices from the coupling_matrices/ directory, and recompute them if not (using multithreading to speed up the 9j manipulations)
# This can handle both odd and even parity NPCFs.
# The output is saved to the working directory with the same format as the NPCF counts, with the filename ...zeta_{N}pcf.txt

import sys, os
import subprocess
import numpy as np
import multiprocessing
from sympy.physics.wigner import wigner_3j, wigner_9j

## First read-in the input file string from the command line
if len(sys.argv)!=5:
    raise Exception("Need to specify the input files, N_data, N_randoms and N_threads!")
else:
    inputs = str(sys.argv[1])
    Ndata = int(sys.argv[2])
    Nrandoms = int(sys.argv[3])
    threads = int(sys.argv[4])

print("Reading in files starting with %s\n"%inputs)

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

# Decide which N we're using
Ns = []
for N in range(10):
    R_file = inputs+'.r0_%dpcf.txt'%N
    if os.path.exists(R_file):
        Ns.append(N)

if len(Ns)==0:
    raise Exception("No files found with input string %s"%inputs)

for N in Ns:
    countsR_all = []
    for i in range(Nrandoms):
        # First load in R pieces
        R_file = inputs+'.r%d_%dpcf.txt' % (i, N)
        if N>2:
            countsR = np.loadtxt(R_file,skiprows=5+N) # skipping rows with radial bins
        else:
            countsR = np.loadtxt(R_file,skiprows=5)

        # Extract ells and radial bins
        if N==2:
            bin1 = np.loadtxt(R_file,skiprows=4,max_rows=1)
        elif N==3:
            ell_1 = np.asarray(countsR[:,0],dtype=int)
            max_ell = np.max(ell_1)
            countsR = countsR[:,1:]
            bin1,bin2 = np.loadtxt(R_file,skiprows=6,max_rows=2)
        elif N==4:
            ell_1,ell_2,ell_3 = np.asarray(countsR[:,:3],dtype=int).T
            max_ell = np.max(ell_1)
            countsR = countsR[:,3:]
            bin1,bin2,bin3 = np.loadtxt(R_file,skiprows=6,max_rows=3)
        elif N==5:
            ell_1,ell_2,ell_12,ell_3,ell_4 = np.asarray(countsR[:,:5],dtype=int).T
            max_ell = np.max(ell_1)
            countsR = countsR[:,5:]
            bin1,bin2,bin3,bin4 = np.loadtxt(R_file,skiprows=6,max_rows=4)
        elif N==6:
            ell_1,ell_2,ell_12,ell_3,ell_123,ell_4,ell_5 = np.asarray(countsR[:,:7],dtype=int).T
            max_ell = np.max(ell_1)
            countsR = countsR[:,7:]
            bin1,bin2,bin3,bin4,bin5 = np.loadtxt(R_file,skiprows=6,max_rows=5)
        else:
            raise Exception("%dPCF not yet configured"%N)
        # save R counts piece
        countsR_all.append(countsR)
    countsR_all = np.asarray(countsR_all)
    countsR = np.mean(countsR_all, axis=0)
    # save filename for later usage
    R_file = inputs+'.r0_%dpcf.txt'%N

    countsN_alldata = []
    for j in range(Ndata+1): # extra iteration to average over data files
        if j < Ndata:
            # Now load in D-R pieces and average
            countsN_all = []
            total_DmR = 0
            for i in range(Nrandoms):
                DmR_file = inputs+'.%d.n%d_%dpcf.txt' % (j, i, N)
                if not os.path.exists(DmR_file): continue
                # Extract counts
                if N==2:
                    countsN_all.append(np.loadtxt(DmR_file, skiprows=5))
                elif N==3:
                    countsN_all.append(np.loadtxt(DmR_file, skiprows=8)[:,1:]) # skipping rows with radial bins and ell
                elif N==4:
                    countsN_all.append(np.loadtxt(DmR_file, skiprows=9)[:,3:])
                elif N==5:
                    countsN_all.append(np.loadtxt(DmR_file, skiprows=10)[:,5:])
                elif N==6:
                    countsN_all.append(np.loadtxt(DmR_file, skiprows=11)[:,7:])
            countsN_all = np.asarray(countsN_all)
            countsN = np.mean(countsN_all, axis=0)
            countsN_alldata.append(countsN)
        else: # do average among all data in the last iteration
            countsN_alldata = np.asarray(countsN_alldata)
            countsN = np.mean(countsN_alldata, axis=0)
        # could use this next line to compute std from finite number of randoms!
        #countsNsig = np.std(countsN_all,axis=0)/np.sqrt(Nrandoms)

        # define zeta filename here as it's similar for all cases
        zeta_file = inputs+'.%d.zeta_%dpcf.txt' % (j, N)
        if j == Ndata: # if we average over data
            zeta_file = inputs+'.zeta_%dpcf.txt' % N

        # Now compute edge-correction equations

        if N==2:
            # isotropic 2PCF is easy!
            zeta = countsN/countsR

            # Now save the output to file, copying the first few lines from the N files
            rfile = open(R_file,"r")
            zfile = open(zeta_file,"w")
            for l,line in enumerate(rfile):
                if l>=5: continue
                zfile.write(line)
            for a in range(len(zeta)):
                zfile.write("%.8e\t"%zeta[a])

        if N==3:
            # Define coupling coefficients, rescaling by R_{ell=0}
            assert ell_1[0]==0
            f_ell = countsR/countsR[0] # (first row should be unity!)

            # Load coupling matrix
            LMAX = max(ell_1)
            input_weights = get_script_path()+'/../coupling_matrices/edge_correction_matrix_%dpcf_LMAX%d.npy'%(N,LMAX)
            if os.path.exists(input_weights):
                print("Loading edge correction weights from file.")
            else:
                # Compute weights from scratch
                subprocess.call(["python",get_script_path()+"/edge_correction_weights.py","%d"%N,"%d"%LMAX,'%d'%threads])

            coupling_tmp = np.load(input_weights)

            # Define coupling matrix
            coupling_matrix = np.zeros((len(ell_1),len(ell_1),len(bin1)))
            for i in range(len(ell_1)):
                for j in range(len(ell_1)):
                    for k in range(len(ell_1)):
                        coupling_matrix[i,j] += coupling_tmp[i,j,k]*f_ell[k]

            ## Now invert matrix equation to get zeta
            # Note that our matrix definition is symmetric
            zeta = np.zeros_like(countsN)
            for i in range(len(bin1)):
                zeta[:,i] = np.matmul(np.linalg.inv(coupling_matrix[:,:,i]),countsN[:,i]/countsR[0,i])

            # Now save the output to file, copying the first few lines from the N files
            rfile = open(R_file,"r")
            zfile = open(zeta_file,"w")
            for l,line in enumerate(rfile):
                if l>=8: continue
                zfile.write(line)
            for a in range(len(zeta)):
                zfile.write("%d\t"%ell_1[a])
                for b in range(len(zeta[a])):
                    zfile.write("%.8e\t"%zeta[a,b])
                zfile.write("\n")
            zfile.close()

        if N==4:
            # Define coupling coefficients, rescaling by R_{Lambda=0}
            assert ell_1[0]==ell_2[0]==ell_3[0]
            f_Lambda = countsR/countsR[0] # (first row should be unity!)

            # Decide if we're using all-parity or even-parity multiplets
            if(np.sum((-1)**(ell_1+ell_2+ell_3)==1)==len(ell_1)):
                # no odd-parity contributions!
                all_parity=0
            else:
                # contains odd-parity terms!
                all_parity=1

            # Load coupling matrix
            LMAX = max(ell_1)
            if all_parity:
                input_weights = get_script_path()+'/../coupling_matrices/edge_correction_matrix_%dpcf_LMAX%d_all.npy'%(N,LMAX)
            else:
                input_weights = get_script_path()+'/../coupling_matrices/edge_correction_matrix_%dpcf_LMAX%d.npy'%(N,LMAX)
            if os.path.exists(input_weights):
                print("Loading edge correction weights from file.")
            else:
                # Compute weights from scratch
                subprocess.run(["python",get_script_path()+"/edge_correction_weights.py","%d"%N,"%d"%LMAX,'%d'%threads,'%d'%all_parity])

            coupling_tmp = np.load(input_weights)

            # Define coupling matrix, by iterating over all Lambda triples
            coupling_matrix = np.zeros((len(ell_1),len(ell_1),len(bin1)))
            for i in range(len(ell_1)):
                for j in range(len(ell_1)):
                    for k in range(len(ell_1)):
                        coupling_matrix[i,j] += coupling_tmp[i,j,k]*f_Lambda[k]

            ## Now invert matrix equation to get zeta
            # Note that our matrix definition is symmetric
            zeta = np.zeros_like(countsN)
            for i in range(len(bin1)):
                zeta[:,i] = np.matmul(np.linalg.inv(coupling_matrix[:,:,i]),countsN[:,i]/countsR[0,i])

            # Now save the output to file, copying the first few lines from the N files
            rfile = open(R_file,"r")
            zfile = open(zeta_file,"w")
            for l,line in enumerate(rfile):
                if l>=9: continue
                zfile.write(line)
            for a in range(len(zeta)):
                zfile.write("%d\t"%ell_1[a])
                zfile.write("%d\t"%ell_2[a])
                zfile.write("%d\t"%ell_3[a])
                for b in range(len(zeta[a])):
                    zfile.write("%.8e\t"%zeta[a,b])
                zfile.write("\n")
            zfile.close()

        if N==5:
            # Define coupling coefficients, rescaling by R_{Lambda=0}
            assert ell_1[0]==ell_2[0]==ell_3[0]==ell_12[0]==ell_4[0]
            f_Lambda = countsR/countsR[0] # (first row should be unity!)

            # Decide if we're using all-parity or even-parity multiplets
            if(np.sum((-1)**(ell_1+ell_2+ell_3+ell_4)==1)==len(ell_1)):
                # no odd-parity contributions!
                all_parity=0
            else:
                # contains odd-parity terms!
                all_parity=1

            # Load coupling matrix
            LMAX = max(ell_1)
            if all_parity:
                input_weights = get_script_path()+'/../coupling_matrices/edge_correction_matrix_%dpcf_LMAX%d_all.npy'%(N,LMAX)
            else:
                input_weights = get_script_path()+'/../coupling_matrices/edge_correction_matrix_%dpcf_LMAX%d.npy'%(N,LMAX)
            if os.path.exists(input_weights):
                print("Loading edge correction weights from file.")
            else:
                # Compute weights from scratch
                subprocess.run(["python",get_script_path()+"/edge_correction_weights.py","%d"%N,"%d"%LMAX,'%d'%threads,'%d'%all_parity])
            coupling_tmp = np.load(input_weights)

            # Define coupling matrix, by iterating over all Lambda triples
            coupling_matrix = np.zeros((len(ell_1),len(ell_1),len(bin1)))
            for i in range(len(ell_1)):
                for j in range(len(ell_1)):
                    for k in range(len(ell_1)):
                        coupling_matrix[i,j] += coupling_tmp[i,j,k]*f_Lambda[k]

            ## Now invert matrix equation to get zeta
            # Note that our matrix definition is symmetric
            zeta = np.zeros_like(countsN)
            for i in range(len(bin1)):
                zeta[:,i] = np.matmul(np.linalg.inv(coupling_matrix[:,:,i]),countsN[:,i]/countsR[0,i])

            # Now save the output to file, copying the first few lines from the N files
            rfile = open(R_file,"r")
            zfile = open(zeta_file,"w")
            for l,line in enumerate(rfile):
                if l>=10: continue
                zfile.write(line)
            for a in range(len(zeta)):
                zfile.write("%d\t"%ell_1[a])
                zfile.write("%d\t"%ell_2[a])
                zfile.write("%d\t"%ell_12[a])
                zfile.write("%d\t"%ell_3[a])
                zfile.write("%d\t"%ell_4[a])
                for b in range(len(zeta[a])):
                    zfile.write("%.8e\t"%zeta[a,b])
                zfile.write("\n")
            zfile.close()

        if N==6:
            # Define coupling coefficients, rescaling by R_{Lambda=0}
            assert ell_1[0]==ell_2[0]==ell_12[0]==ell_3[0]==ell_123[0]==ell_4[0]==ell_5[0]
            f_Lambda = countsR/countsR[0] # (first row should be unity!)

            # Decide if we're using all-parity or even-parity multiplets
            if(np.sum((-1)**(ell_1+ell_2+ell_3+ell_4+ell_5)==1)==len(ell_1)):
                # no odd-parity contributions!
                all_parity=0
            else:
                # contains odd-parity terms!
                all_parity=1

            # Load coupling matrix
            LMAX = max(ell_1)
            if all_parity:
                input_weights = get_script_path()+'/../coupling_matrices/edge_correction_matrix_%dpcf_LMAX%d_all.npy'%(N,LMAX)
            else:
                input_weights = get_script_path()+'/../coupling_matrices/edge_correction_matrix_%dpcf_LMAX%d.npy'%(N,LMAX)
            if os.path.exists(input_weights):
                print("Loading edge correction weights from file.")
            else:
                # Compute weights from scratch
                subprocess.run(["python",get_script_path()+"/edge_correction_weights.py","%d"%N,"%d"%LMAX,'%d'%threads,'%d'%all_parity])
            coupling_tmp = np.load(input_weights)

            # Define coupling matrix, by iterating over all Lambda triples
            coupling_matrix = np.zeros((len(ell_1),len(ell_1),len(bin1)))
            for i in range(len(ell_1)):
                for j in range(len(ell_1)):
                    for k in range(len(ell_1)):
                        coupling_matrix[i,j] += coupling_tmp[i,j,k]*f_Lambda[k]

            ## Now invert matrix equation to get zeta
            # Note that our matrix definition is symmetric
            print("Coupling matrix computed")
            zeta = np.zeros_like(countsN)
            for i in range(len(bin1)):
                zeta[:,i] = np.matmul(np.linalg.inv(coupling_matrix[:,:,i]),countsN[:,i]/countsR[0,i])

            # Now save the output to file, copying the first few lines from the N files
            rfile = open(R_file,"r")
            zfile = open(zeta_file,"w")
            for l,line in enumerate(rfile):
                if l>=11: continue
                zfile.write(line)
            for a in range(len(zeta)):
                zfile.write("%d\t"%ell_1[a])
                zfile.write("%d\t"%ell_2[a])
                zfile.write("%d\t"%ell_12[a])
                zfile.write("%d\t"%ell_3[a])
                zfile.write("%d\t"%ell_123[a])
                zfile.write("%d\t"%ell_4[a])
                zfile.write("%d\t"%ell_5[a])
                for b in range(len(zeta[a])):
                    zfile.write("%.8e\t"%zeta[a,b])
                zfile.write("\n")
            zfile.close()

        if j < Ndata:
            print("Computed %dPCF using %d (random and data-random) files, saving to %s\n" % (N, Nrandoms, zeta_file))
        else:
            print("Averaged %dPCF using %d (data) files, saving to %s\n" % (N, Ndata, zeta_file))
