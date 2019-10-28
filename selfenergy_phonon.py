# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 01:44:04 2016

@author: Ben
"""

#
# Warnings!!!
# upper and lower case MPI commands? mpi data types?
# different python versions for einsum and mpi4py????
# why can't I reduce a complex number array??
#

import subprocess
#bash_command('source mlpython2.7.5.sh')
    
import numpy as np
#from scipy import linalg
import time
import sys, os

#from functions import *

savedir   = sys.argv[1]

myrank=0
nprocs=1

def bash_command(cmd):
    subprocess.Popen(['/bin/bash', '-c', cmd])

if not os.path.exists(savedir+'dataPhonon/'):
    os.mkdir(savedir+'dataPhonon/')
 
def parseline(mystr):
    ind = mystr.index('#')
    return mystr[ind+1:]
       
with open(savedir+'input','r') as f:
    Nt    = int(parseline(f.readline()))
    Ntau  = int(parseline(f.readline()))
    dt    = float(parseline(f.readline()))
    dtau  = float(parseline(f.readline()))
    Nkx   = int(parseline(f.readline()))
    Nky   = int(parseline(f.readline()))
    g2    = float(parseline(f.readline()))
    omega = float(parseline(f.readline()))    
    pump  = int(parseline(f.readline()))

Norbs = 2
    
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


startTime = time.time()


option = 'G'

if option=='S':
    S = np.load(savedir+'Sdir/SG.npy') - np.load(savedir+'Sdir/SL.npy')
else:
    S = np.load(savedir+'Glocdir/GlocG.npy') - np.load(savedir+'Glocdir/GlocL.npy')

GL = np.load(savedir+'Glocdir/GlocL.npy')
dens = np.zeros(Nt, dtype=complex)
for it in range(Nt):
    for ib in range(Norbs):
        dens[it] += GL[ib*Nt+it, ib*Nt+it]/(Nkx*Nky)

np.save(savedir+'dataPhonon/dens.npy', dens)
print 'dens'
print 'shape', np.shape(dens)
print dens

exit()


Nw = len(range(Nt/2, Nt))
ts = np.linspace(0, (Nw-1)*2*dt, Nt)

SRt = np.zeros(Nw, dtype=complex)
SRw = np.zeros(Nw, dtype=complex)

for it in range(Nt/2, Nt):
    for ib in range(Norbs):
        SRt[it-Nt/2] += S[ib*Nt+it, ib*Nt + Nt - it - 1]
 
fftS = np.fft.fft(SRt) * 2*dt
freqs = np.fft.fftfreq(Nw, 2*dt)
fftS[0] = 0.

#print fftS[0:10]

freqs, fftS = zip(*sorted(zip(freqs, fftS)))
freqs = 2*np.pi*np.array(freqs)

SRw = fftS

if option=='S':
    np.save(savedir+'dataPhonon/SRw.npy', SRw)  
else:
    np.save(savedir+'dataPhonon/DOSw.npy', SRw)  

np.save(savedir+'dataPhonon/freqs.npy', freqs)


    

    
