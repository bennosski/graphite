# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 01:44:04 2016

@author: Ben
"""

#
# Warnings!!!
#
# fix band / time indices for ARPES. THIS HAS CHANGED!!
#

import subprocess
#bash_command('source mlpython2.7.5.sh')
    
import numpy as np
#from scipy import linalg
import time
import sys, os
from functions import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
nprocs = comm.size
myrank = comm.rank

def bash_command(cmd):
    subprocess.Popen(['/bin/bash', '-c', cmd])

Norbs = 2        


#inputfile = sys.argv[1]
savedir   = sys.argv[1]
nprocs_original = int(sys.argv[2])
LG = sys.argv[3]

debugdir = savedir + 'debugARPES/'

if myrank==0:
    if not os.path.exists(debugdir):
        os.mkdir(debugdir)
comm.barrier()

myfile = open(debugdir+"log%d"%myrank, 'w')

if myrank==0:
    print ' '
    print 'nprocs = ',nprocs
    myfile.write('nprocs = %d'%nprocs)

comm.barrier()
        
with open(savedir+'input','r') as f:
    Nt   = int(parseline(f.readline()))
    Ntau = int(parseline(f.readline()))
    dt   = float(parseline(f.readline()))
    dtau = float(parseline(f.readline()))
    Nkx  = int(parseline(f.readline()))
    Nky  = int(parseline(f.readline()))

if myrank==0:
    print '\n','Params'
    print 'Nt    = ',Nt
    print 'Ntau  = ',Ntau
    print 'dt    = ',dt
    print 'dtau  = ',dtau
    print 'Nky   = ',Nkx
    print 'Nky   = ',Nky
    print '\n'


if myrank==0:
    print 'nprocs_original ', nprocs_original
    
if myrank==0:
    startTime = time.time()
    


# need to setup cut coordinates and indices
# setup kpp, k2p, k2i, i2k for the new cut






 
#2.4 is somewhat broad spectra in energy, but ok
#probe_width = 2.4  # time units
#probe_width = 5.0  # time units
probe_width = 3.0 # worked well
Nw = 180

times = np.linspace(0, (Nt-1)*dt, Nt)

time_indices = []
for it, t in enumerate(times):
    if t>probe_width*2.0 and t<(Nt-1)*dt - probe_width*2.0:
        if len(time_indices)==0:
            time_indices.append(it)
        elif t - times[time_indices[-1]] > probe_width/4:
            time_indices.append(it)

if myrank==0:
    print 'time indices. num = ',len(time_indices)
    print time_indices

Nt_arpes = len(time_indices)
#I = np.zeros([3, kpp, Nw, Nt_arpes])
I = np.zeros([Nkpoints, Nw, Nt_arpes])

def init_probe(Nt, dt, it0, probe_width):
    t0 = it0*dt
    p = np.zeros([Nt, Nt])
    for it1 in range(Nt):
        t1 = dt*it1
        for it2 in range(Nt):
            t2 = dt*it2
            p[it1,it2] = 1./(2.*np.pi*probe_width**2) * np.exp(-(t1-t0)**2/(2.*probe_width**2)  - (t2-t0)**2/(2.*probe_width**2))
    return p

tmtp = np.zeros([Nt,Nt], dtype=complex)
for it1 in range(Nt):
    for it2 in range(Nt):
        tmtp[it1,it2] = (it1-it2)*dt

if myrank==0:
    print 'initialization time ',time.time()-startTime
    startTime = time.time()

myfile.write('done initialize')


# split time indices based on rank%(Nkpoints)
Nti = len(time_indices)            

for index,it0 in enumerate(time_indices):


    #savename = savedir+'ARPES'+LG+'/I%d'%ib+'_%d'%cut_nums[ik]+'_%d'%index+'.npy'

    if myrank==0:
        print 'working on it0',it0,' number ',index,' out of ',len(time_indices)

    myfile.write('working on it0 %d'%it0+' number %d'%index+' out of %d'%len(time_indices))
  
    p = init_probe(Nt, dt, it0, probe_width)
                   
    
    for ik in range(kpp):

        # dividing up time slices on the processors
        if index%ppk[cut_nums[ik]]==myrank/Nkpoints: # time index mod num processors working on this k point == process assigned to this time point (we know it is assigned to this k point)

            ik1, ik2 = i2k[cut_inds[ik]]
            
            GLess = np.zeros([Nt, Nt], dtype=complex)
            for it1 in range(Nt):
                for it2 in range(Nt):

                    G0Lk = compute_G0L(ik1, ik2, myrank, Nkx, Nky, kpp, k2p, k2i, Nt, Ntau, dt, dtau, fks, UksR, eks, Norbs)
                    

                    

                    Gorb = np.zeros([Norbs,Norbs], dtype=complex)
                    for ib1 in range(Norbs):
                        for ib2 in range(Norbs):
                            

                            Gorb[ib1,ib2] = G[ik][ib1*Nt+it1, ib2*Nt+it2]

                    GLess[it1, it2] = np.trace(Gorb)


            myfile.write('    working on ik = %d'%ik)

            freqs = np.linspace(-2.0, 2.0, Nw)

            for iw,w in enumerate(freqs):

                myfile.write('            working on iw = %d'%iw)


                #for ib in range(Norbs):
                I[cut_nums[ik],iw,index] = np.imag(np.sum( np.exp(1j*w*tmtp) * p * GLess )) * dt**2


                
    if myrank==0:
        print 'iteration end time',time.time()-startTime


Itotal = np.zeros([Nkpoints, Nw, Nt_arpes])
comm.Reduce(I, Itotal, op=MPI.SUM, root=0)

if myrank==0:
    print 'finished reduction and saving',time.time()-startTime
    np.save(savedir+'ARPES'+LG+'/I.npy', Itotal)
   
if myrank==0:
    print 'done'

myfile.close()

MPI.Finalize()
    

    
