
import os, sys
import numpy as np
from functions import *

savedir = sys.argv[1]

Nt    = 3000
Ntau  = 100
Norbs = 2

Sigma = langreth(Nt, Ntau, Norbs)
Sigma.myload(savedir+'Sdir/S')

EESigma = langreth(Nt, Ntau, Norbs)
EESigma.myload(savedir+'EESigma/EESigma')
#EESigma.scale(1.0/120**6) # 3 copies, 2D lattice = 6 in exponent

print 'max EESigma'
print np.amax(EESigma.G)
print np.amax(EESigma.L)
print np.amax(EESigma.RI)
print np.amax(EESigma.IR)
print np.amax(EESigma.M)

print 'mean abs EESigma'
print np.mean(np.abs(EESigma.G))
print np.mean(np.abs(EESigma.L))
print np.mean(np.abs(EESigma.RI))
print np.mean(np.abs(EESigma.IR))
print np.mean(np.abs(EESigma.M))


print 'max Sigma'
print np.amax(Sigma.G)
print np.amax(Sigma.L)
print np.amax(Sigma.RI)
print np.amax(Sigma.IR)
print np.amax(Sigma.M)


print 'mean abs Sigma'
print np.mean(np.abs(Sigma.G))
print np.mean(np.abs(Sigma.L))
print np.mean(np.abs(Sigma.RI))
print np.mean(np.abs(Sigma.IR))
print np.mean(np.abs(Sigma.M))


