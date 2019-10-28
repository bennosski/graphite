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
import resource
import collect_ARPES_data as cAd


comm = MPI.COMM_WORLD
nprocs = comm.size
myrank = comm.rank

def bash_command(cmd):
    subprocess.Popen(['/bin/bash', '-c', cmd])

Norbs = 2        

#inputfile = sys.argv[1]
savedir   = sys.argv[1]
LG = sys.argv[2]

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

with open(savedir+'input', 'r') as f:
    Nt    = int(parseline(f.readline()))
    Ntau  = int(parseline(f.readline()))
    dt    = float(parseline(f.readline()))
    dtau  = float(parseline(f.readline()))
    Nkx   = int(parseline(f.readline()))
    Nky   = int(parseline(f.readline()))
    g2    = float(parseline(f.readline()))
    omega = float(parseline(f.readline()))    
    pump  = int(parseline(f.readline()))
    
if myrank==0:
    print '\n','Params'
    print 'Nt    = ',Nt
    print 'Ntau  = ',Ntau
    print 'dt    = ',dt
    print 'dtau  = ',dtau
    print 'Nkx   = ',Nkx
    print 'Nky   = ',Nky
    print 'g2    = ',g2
    print 'omega = ',omega
    print 'pump  = ',pump
    print '\n'

    
if myrank==0:
    startTime = time.time()
    
#Sigma = langreth(Nt, Ntau, Norbs)
#Sigma.myload(savedir+'Sdir/S')
#Sigma.scale(1.0);

# need to setup cut coordinates and indices
# setup kpp, k2p, k2i, i2k for the new cut
# init Uks for these points

#Nk = 101 # number of points on the cut
Nk = 96

k2p, k2i, i2k = init_k2p_k2i_i2k(Nk, 1, nprocs, myrank)
kpp = np.count_nonzero(k2p==myrank)
#ppk = nprocs//Nk


'''
if myrank==0:
    print 'kx, ky'
    for ik in range(Nk):
        kx,ky = get_kx_ky_ARPES(ik,Nk)
        print kx, ky
'''
#comm.barrier()
#exit()

UksR, UksI, eks, fks = init_Uks_ARPES(myrank, Nk, kpp, k2p, k2i, Nt, Ntau, dt, dtau, pump, Norbs)
 
#2.4 is somewhat broad spectra in energy, but ok
#probe_width = 2.4  # time units
#probe_width = 5.0  # time units
#probe_width = 3.0 # worked well
probe_width = 10.0

Nw = 100

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
I = np.zeros([Nk, Nw, Nt_arpes])

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

# check for existing data
counts = cAd.get_counts(savedir+'ARPES'+LG+'/')


# split time indices based on rank%(Nkpoints)
Nti = len(time_indices)            
                   
for ik in range(Nk):    
    
    # dividing up time slices on the processors
    #if index%ppk[cut_nums[ik]]==myrank/Nk: # time index mod num processors working on this k point == process assigned to this time point (we know it is assigned to this k point)
    if myrank==k2p[ik,0]:

        if myrank==0:
            print 'inside for ik',ik,'rank = ',myrank
        
        GLess = np.zeros([Nt, Nt], dtype=complex)
        G0k = compute_G0_ARPES(ik, myrank, Nk, kpp, k2p, k2i, Nt, Ntau, dt, dtau, fks, UksR, UksI, eks, Norbs)

        '''
        if myrank==0:
            print 'Memory usage after G0k: %s (kb)'% resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        temp = multiply(G0k, Sigma, Nt, Ntau, dt, dtau, Norbs)
        temp.scale(-1.0)

        # i think the integral done by multiply means temp has no delta piece
        # we add a delta piece to add the identity in I - G0*Sigma
        temp.DR = np.ones(Norbs*Nt) / dt
        temp.DM = np.ones(Norbs*Ntau) / (-1j*dtau)

        # copies are good so that the diagonals of the langreth matrices don't get changed during solve
        Gk = solve(temp, G0k, Nt, Ntau, dt, dtau, Norbs)

        if myrank==0:
            print 'Memory usage after Gk: %s (kb)'% resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        '''

        for it1 in range(Nt):
            for it2 in range(Nt):
                Gorb = np.zeros([Norbs,Norbs], dtype=complex)
                for ib1 in range(Norbs):
                    for ib2 in range(Norbs):
                        Gorb[ib1,ib2] = G0k.L[ib1*Nt+it1, ib2*Nt+it2]
                GLess[it1, it2] = np.trace(Gorb)

        myfile.write('    working on ik = %d'%ik)

        if myrank==nprocs-1:
            print 'done Gless'

        freqs = np.linspace(-1.0, 1.0, Nw)

        # split this based on myrank as well
        for index,it0 in enumerate(time_indices):
        
            #if index>0:
            #    break
            
            if ik in counts and index<counts[ik]:
                continue


            if myrank==0:
                print 'working on it0',it0,' number ',index,' out of ',len(time_indices)

            p = init_probe(Nt, dt, it0, probe_width)

            pGLess = p * GLess

            # cut out region around it0
            ipw = round(probe_width/dt)
            i1 = int(it0-2*ipw)
            i2 = int(it0+2*ipw)
            mytimes = tmtp[i1:i2, i1:i2]
            pGLess  = pGLess[i1:i2, i1:i2] * dt**2

            for iw,w in enumerate(freqs):
                I[ik, iw, index] = np.imag(np.sum(np.exp(1j*w*mytimes) * pGLess))                
                #I[ik,iw,index] = np.imag(np.sum( np.exp(1j*w*tmtp) * p * GLess )) * dt**2

            if myrank==0:
                print 'iteration end time',time.time()-startTime

            np.save(savedir+'ARPES'+LG+'/I%d_%d'%(ik,index)+'.npy', I[ik,:,index])


'''
Itotal = np.zeros([Nk, Nw, Nt_arpes])
comm.Reduce(I, Itotal, op=MPI.SUM, root=0)

if myrank==0:
    print 'finished reduction and saving',time.time()-startTime
    np.save(savedir+'ARPES'+LG+'/I.npy', Itotal)
'''
   
if myrank==0:
    print 'done'

myfile.close()

MPI.Finalize()
    

    
