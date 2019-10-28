# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 01:44:04 2016

@author: Ben
"""
#
# computes e-e selfenergy after a run has completed computing the local Green's function
#


import subprocess
    
import numpy as np
import time
import sys, os
from functions import *

# the directory from which to load
savedir   = sys.argv[1]

myrank=0
nprocs=1

def bash_command(cmd):
    subprocess.Popen(['/bin/bash', '-c', cmd])

if not os.path.exists(savedir+'EESigma/'):
    os.mkdir(savedir+'EESigma/')
       
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

Gloc = langreth(Nt, Ntau, Norbs)
Gloc.myload(savedir+'Glocdir/Gloc')
Gloc.scale(1.0/(Nkx*Nky))

print 'done load Gloc ',time.time()-startTime
startTime = time.time()

X = langreth(Nt, Ntau, Norbs)
X.mycopy(Gloc)

print 'done copy ',time.time()-startTime
startTime = time.time()

X.transpose()

print 'done transpose ',time.time()-startTime
startTime = time.time()

X.directMultiply(Gloc)
X.directMultiply(Gloc)

print 'done mults ',time.time()-startTime
startTime = time.time()

#np.save(savedir+'EESigma/EESigma.npy', X);
X.mysave(savedir+'EESigma/EESigma')




    

    
