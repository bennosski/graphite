import numpy as np
#from mpi4py import MPI
from scipy.linalg import expm

def parseline(mystr):
    ind = mystr.index('#')
    return mystr[ind+1:]


class langreth:
    # would be easier to have Nt, Ntau, Norbs as member variables

    def __init__(self, Nt, Ntau, Norbs):
        self.G  = np.zeros([Norbs*Nt, Norbs*Nt], dtype=complex)
        self.L  = np.zeros([Norbs*Nt, Norbs*Nt], dtype=complex)
        self.IR = np.zeros([Norbs*Ntau, Norbs*Nt], dtype=complex)
        self.RI = np.zeros([Norbs*Nt, Norbs*Ntau], dtype=complex)
        self.M  = np.zeros([Norbs*Ntau, Norbs*Ntau], dtype=complex)
        self.DR = np.zeros(Norbs*Nt, dtype=complex)
        self.DM = np.zeros(Norbs*Ntau, dtype=complex)

    def add(self, b):
        self.G  += b.G
        self.L  += b.L
        self.IR += b.IR
        self.RI += b.RI
        self.M  += b.M

    def directMultiply(self, b):
        self.G  *= b.G
        self.L  *= b.L
        self.IR *= b.IR
        self.RI *= b.RI
        self.M  *= b.M
        #self.initRA(Nt)
        
    def scale(self, c):
        self.G  *= c
        self.L  *= c
        self.IR *= c
        self.RI *= c
        self.M  *= c
        
    def mycopy(self, b):
        self.G  = b.G.copy()
        self.L  = b.L.copy()
        self.IR = b.IR.copy()
        self.RI = b.RI.copy()
        self.M  = b.M.copy()
        self.DR = b.DR.copy()
        self.DM = b.DM.copy()        
        
    def transpose(self):
        Gt = np.transpose(self.G)
        Lt = np.transpose(self.L)
        IRt = np.transpose(self.IR)
        RIt = np.transpose(self.RI)
        # note the proper switching between components
        self.G = Lt     
        self.L = Gt
        self.IR = RIt
        self.RI = IRt
        self.M = np.transpose(self.M)
        
    def zero(self, Nt, Ntau, Norbs):
        self.G  = np.zeros([Norbs*Nt, Norbs*Nt], dtype=complex)
        self.L  = np.zeros([Norbs*Nt, Norbs*Nt], dtype=complex)
        self.IR = np.zeros([Norbs*Ntau, Norbs*Nt], dtype=complex)
        self.RI = np.zeros([Norbs*Nt, Norbs*Ntau], dtype=complex)
        self.M  = np.zeros([Norbs*Ntau, Norbs*Ntau], dtype=complex)
        self.DR = np.zeros(Norbs*Nt, dtype=complex)
        self.DM = np.zeros(Norbs*Ntau, dtype=complex)

    def mysave_old(self, myfile, Nt, Ntau, myrank, ik):
        '''
        arr = np.zeros([3*2*Nt+3*Ntau, 3*2*Nt+3*Ntau], dtype=complex)
        arr[0:3*Nt,               0:3*Nt]   = self.R
        arr[3*Nt:3*2*Nt  ,        0:3*Nt]   = self.G
        arr[0:3*Nt,          3*Nt:3*2*Nt]   = self.L
        arr[3*Nt:3*2*Nt,     3*Nt:3*2*Nt]   = self.A
        arr[0:3*Nt, 3*2*Nt:3*2*Nt+3*Ntau]   = self.RI
        arr[3*2*Nt:3*2*Nt+3*Ntau, 0:3*Nt]   = self.IR
        arr[3*2*Nt:3*2*Nt+3*Ntau, 3*2*Nt:3*2*Nt+3*Ntau] = self.M
        '''
        
        mystr = '%d'%myrank+'_%d'%ik+'.npy'
        np.save(myfile+'G'+mystr, self.G)
        np.save(myfile+'L'+mystr, self.L)
        np.save(myfile+'RI'+mystr, self.RI)
        np.save(myfile+'IR'+mystr, self.IR)
        np.save(myfile+'M'+mystr, self.M)
        np.save(myfile+'DR'+mystr, self.DR)
        np.save(myfile+'DM'+mystr, self.DM)
        
    def mysave(self, myfile):
        np.save(myfile+'G', self.G)
        np.save(myfile+'L', self.L)
        np.save(myfile+'RI', self.RI)
        np.save(myfile+'IR', self.IR)
        np.save(myfile+'M', self.M)
        np.save(myfile+'DR', self.DR)
        np.save(myfile+'DM', self.DM)

    def myload(self, myfile):
        self.G  = np.load(myfile+'G.npy')
        self.L  = np.load(myfile+'L.npy')
        self.RI = np.load(myfile+'RI.npy')
        self.IR = np.load(myfile+'IR.npy')
        self.M  = np.load(myfile+'M.npy')
        
    def myload_old(self, myfile):
        '''
        arr =  np.load(myfile)
        self.R  = arr[0:3*Nt,               0:3*Nt]            
        self.G  = arr[3*Nt:3*2*Nt,          0:3*Nt]
        self.L  = arr[0:3*Nt,          3*Nt:3*2*Nt]       
        self.A  = arr[3*Nt:3*2*Nt,     3*Nt:3*2*Nt]    
        self.RI = arr[0:3*Nt, 3*2*Nt:3*2*Nt+3*Ntau]  
        self.IR = arr[3*2*Nt:3*2*Nt+3*Ntau, 0:3*Nt]  
        self.M  = arr[3*2*Nt:3*2*Nt+3*Ntau, 3*2*Nt:3*2*Nt+3*Ntau]                    
        '''
        mystr = '%d'%myrank+'_%d'%ik+'.npy'
        self.G  = np.load(myfile+'G'+mystr)
        self.L  = np.load(myfile+'L'+mystr)
        self.RI = np.load(myfile+'RI'+mystr)
        self.IR = np.load(myfile+'IR'+mystr)
        self.M  = np.load(myfile+'M'+mystr)


def setup_cuts(Nk):

    if False:
        # gamma X M gamma
        cut_ikxs = []
        cut_ikys = []
        for i in range(Nk//3):
            cut_ikxs.append(i)
            cut_ikys.append(i)
        for i in range(Nk//6):
            cut_ikxs.append(Nk//3+i)
            cut_ikys.append(Nk//3-2*i)
        for i in range(Nk//2+1):
            cut_ikxs.append(Nk//2-i)
            cut_ikys.append(0)

    if True:
        # gamma X
        cut_ikxs = []
        cut_ikys = []
        for i in range(Nk//2):
            cut_ikxs.append(i)
            cut_ikys.append(i)

    return cut_ikxs, cut_ikys

# kpoint on the y axis
def Hk(kx, ky):
    mat = np.zeros([2,2], dtype=complex)
    gammak = 1 + np.exp(1j*kx*np.sqrt(3.)) + np.exp(1j*np.sqrt(3)/2*(kx + np.sqrt(3)*ky))
    mat[0,1] = gammak*2.8
    mat[1,0] = np.conj(gammak)*2.8
    return mat

# k point on the y axis
# returns the positive eigenvalue!
def band(kx, ky):
    #return 2.8 * sqrt(1. + 4*cos(sqrt(3.)/2*kx)*cos(ky/2) + 4*cos(ky/2)**2)
    #return 2.8 * np.sqrt(1. + 4*np.cos(3.0/2*kx)*np.cos(np.sqrt(3)*ky/2) + 4*np.cos(np.sqrt(3.)*ky/2)**2)
    return 2.8 * np.sqrt(1. + 4*np.cos(3.0/2*ky)*np.cos(np.sqrt(3)*kx/2) + 4*np.cos(np.sqrt(3.)*kx/2)**2)

def get_kx_ky(ik1, ik2, Nkx, Nky):
    ky = 4*np.pi/3*ik1/Nkx + 2*np.pi/3*ik2/Nky
    kx = 2*np.pi/np.sqrt(3.)*ik2/Nky
    return kx, ky

def get_kx_ky_ARPES(ik, Nk):
    # cut from 1/4 to 5/12 of the way along the diagonal
    #f = 1./4 + 1./6 * ik/(Nk-1)
    f = (1./4+1./24) + (1./12) * ik/(Nk-1)
    
    # cut along gamma - X
    # ik runs from 0 to Nk/2
    ky = 4*np.pi/3*f + 2*np.pi/3*f
    kx = 2*np.pi/np.sqrt(3.)*f
    return kx, ky


def compute_A(mytime, Nt, dt, pump):
    if pump==0:
        return 0.0, 0.0, 0.0

    if pump==11:
        Amax = 0.5

        fieldAngle = np.pi*150./180.
        cosA    = np.cos(fieldAngle)
        sinA    = np.sin(fieldAngle)

        A = 0.
        if mytime>=18.0 and mytime<=20.0:            
            A =  Amax*np.sin(np.pi/2.*(mytime-18.0))**2
        elif mytime>20.0 and mytime<22.0:
            A = -Amax*np.sin(np.pi/2.*(mytime-20.0))**2

        return A*cosA, A*sinA, fieldAngle

    if pump==1:
        Amax = 0.05

        fieldAngle = np.pi*150./180.
        cosA    = np.cos(fieldAngle)
        sinA    = np.sin(fieldAngle)

        A = 0.
        if mytime>=18.0 and mytime<=20.0:            
            A =  Amax*np.sin(np.pi/2.*(mytime-18.0))**2
        elif mytime>20.0 and mytime<22.0:
            A = -Amax*np.sin(np.pi/2.*(mytime-20.0))**2

        return A*cosA, A*sinA, fieldAngle
    
    if pump==2:
        Amax = 0.05

        fieldAngle = np.pi*150./180.
        cosA    = np.cos(fieldAngle)
        sinA    = np.sin(fieldAngle)

        t1 = 20
        t2 = 124.71975512

        A = 0.
        if mytime>=t1 and mytime<=t2:            
            A = Amax * np.sin(1.2*(mytime-t1)) * np.exp(-(mytime-(t1+t2)/2)**2/40.0**2)

        return A*cosA, A*sinA, fieldAngle

    if pump==3:
        Amax = 0.005

        fieldAngle = np.pi*150./180.
        cosA    = np.cos(fieldAngle)
        sinA    = np.sin(fieldAngle)

        t1 = 40.0
        t2 = t1 + 104.71975512

        A = 0.
        if mytime>=t1 and mytime<=t2:            
            A = Amax * np.sin(1.2*(mytime-t1)) * np.exp(-(mytime-(t1+t2)/2)**2/40.0**2)

        return A*cosA, A*sinA, fieldAngle


    if pump==4:
        Amax = 0.001

        fieldAngle = np.pi*150./180.
        cosA    = np.cos(fieldAngle)
        sinA    = np.sin(fieldAngle)

        t1 = 40.0
        t2 = t1 + 104.71975512

        A = 0.
        if mytime>=t1 and mytime<=t2:            
            A = Amax * np.sin(1.2*(mytime-t1)) * np.exp(-(mytime-(t1+t2)/2)**2/40.0**2)


        return A*cosA, A*sinA, fieldAngle

    
    return None

def init_k2p_k2i_i2k(Nkx, Nky, nprocs, myrank):
    k2p = np.zeros([Nkx, Nky], dtype=int)
    k2i = np.zeros([Nkx, Nky], dtype=int)
    i2k = []
    for ik1 in range(Nkx):
        for ik2 in range(Nky):
            k2p[ik1,ik2] = (ik1*Nky + ik2)%nprocs
            k2i[ik1,ik2] = (ik1*Nky + ik2)//nprocs
            if k2p[ik1,ik2]==myrank:
                i2k.append([ik1,ik2])
    return k2p, k2i, i2k

# note 1 on diagonal. This means that SigmaR is nonzero for t=t' which is good. 
# matches G0 definition
# is this the right choice? 
def init_theta(NT):
    #theta = np.zeros([NT,NT])
    theta = np.diag(0.5 * np.ones(NT))
    for i in range(NT):
        for j in range(i):
            theta[i,j] = 1.0
    #theta = theta + 0.5*np.diag(np.ones(NT))
    return theta

def init_block_theta(theta, Nt, Norbs):
    for a in range(Norbs):
        for b in range(Norbs):
            for i in range(Nt):
                theta[a*Nt+i,b*Nt+i] = 0.5
                for j in range(i):
                    theta[a*Nt+i,b*Nt+j] = 1.0

def init_Uks_ARPES(myrank, Nk, kpp, k2p, k2i, Nt, Ntau, dt, dtau, pump, Norbs):
    
    beta = Ntau*dtau
    
    UksR = np.zeros([kpp, Nt, Norbs, Norbs], dtype=complex)
    UksI = np.zeros([kpp, Ntau, Norbs], dtype=complex)
    fks  = np.zeros([kpp, Norbs], dtype=complex)
    eks  = np.zeros([kpp, Norbs], dtype=complex)

    
    for ik in range(Nk):
        if myrank==k2p[ik,0]:                
                index = k2i[ik,0]

                kx, ky = get_kx_ky_ARPES(ik, Nk)
            
                prod = np.diag(np.ones(2))
                UksR[index,0] = prod.copy()
                for it in range(1,Nt):
                    tt = it*dt # - dt/2.0
                    Ax, Ay, _ = compute_A(tt, Nt, dt, pump)
                    prod = np.dot(expm(-1j*Hk(kx-Ax, ky-Ay)*dt), prod)
                    UksR[index,it] = prod.copy()
                                    
                ek = band(kx, ky)
                fpek = 1.0/(np.exp( beta*ek)+1.0)
                fmek = 1.0/(np.exp(-beta*ek)+1.0)
                fks[index][0] = fpek
                fks[index][1] = fmek
                eks[index][0] =  ek
                eks[index][1] = -ek

                # better way since all Hk commute at t=0
                # also pull R across the U(tau,0) so that we work with diagonal things
                for it in range(Ntau):
                    UksI[index,it,0] = np.exp(-ek*dtau*it)
                    UksI[index,it,1] = np.exp(+ek*dtau*it)
                
    return UksR, UksI, eks, fks


                    
def init_Uks(myrank, Nkx, Nky, kpp, k2p, k2i, Nt, Ntau, dt, dtau, pump, Norbs):
    
    beta = Ntau*dtau
    
    UksR = np.zeros([kpp, Nt, Norbs, Norbs], dtype=complex)
    UksI = np.zeros([kpp, Ntau, Norbs], dtype=complex)
    fks  = np.zeros([kpp, Norbs], dtype=complex)
    eks  = np.zeros([kpp, Norbs], dtype=complex)
    
    for ik1 in range(Nkx):
        for ik2 in range(Nky):
            if myrank==k2p[ik1,ik2]:                
                index = k2i[ik1,ik2]

                kx, ky = get_kx_ky(ik1, ik2, Nkx, Nky)
                
                prod = np.diag(np.ones(2))
                UksR[index,0] = prod.copy()
                for it in range(1,Nt):
                    tt = it*dt # - dt/2.0
                    Ax, Ay, _ = compute_A(tt, Nt, dt, pump)
                    prod = np.dot(expm(-1j*Hk(kx-Ax, ky-Ay)*dt), prod)
                    UksR[index,it] = prod.copy()
                                    
                ek = band(kx, ky)
                fpek = 1.0/(np.exp( beta*ek)+1.0)
                fmek = 1.0/(np.exp(-beta*ek)+1.0)
                fks[index][0] = fpek
                fks[index][1] = fmek
                eks[index][0] =  ek
                eks[index][1] = -ek

                # better way since all Hk commute at t=0
                # also pull R across the U(tau,0) so that we work with diagonal things
                for it in range(Ntau):
                    UksI[index,it,0] = np.exp(-ek*dtau*it)
                    UksI[index,it,1] = np.exp(+ek*dtau*it)
                
    return UksR, UksI, eks, fks


def compute_G0(ik1, ik2, myrank, Nkx, Nky, kpp, k2p, k2i, Nt, Ntau, dt, dtau, fks, UksR, UksI, eks, Norbs):
    G0 = langreth(Nt, Ntau, Norbs)
    
    beta  = dtau*Ntau
    theta = init_theta(Ntau)
        
    kx, ky = get_kx_ky(ik1, ik2, Nkx, Nky)

    # check if this is the right k point for this proc
    # this should have been checked before calling this function
    if myrank==k2p[ik1,ik2]:
        index = k2i[ik1,ik2]

        _, R  = np.linalg.eig(Hk(kx, ky))
        G0L = 1j*np.einsum('ij,j,jk->ik', R, fks[index]-0.0, np.conj(R).T) # - 0.0 for lesser Green's function    
        G0G = 1j*np.einsum('ij,j,jk->ik', R, -fks[index]*np.exp(beta*eks[index]), np.conj(R).T) # - 1.0 for greater Green's function    
        for it1 in range(Nt):
            for it2 in range(Nt):
                G = np.einsum('ij,jk,kl->il', UksR[index,it1], G0L, np.conj(UksR[index,it2]).T)
                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.L[a*Nt+it1,b*Nt+it2] = G[a,b]

                G = np.einsum('ij,jk,kl->il', UksR[index,it1], G0G, np.conj(UksR[index,it2]).T)
                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.G[a*Nt+it1,b*Nt+it2] = G[a,b]


        for it1 in range(Nt):
            t1 = it1 * dt
            for it2 in range(Ntau):
                t2 = -1j * it2 * dtau

                UksI_inv = [UksI[index,it2,1], UksI[index,it2,0]]
                G = fks[index] * UksI_inv
                G = np.einsum('ij,jk,kl,lm->im', UksR[index,it1], R, np.diag(G), np.conj(R).T)
                #G = np.einsum('ij,jk,kl->il', UksR[index,it1], G0L,  )
                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.RI[a*Nt+it1,b*Ntau+it2] = G[a,b]

        for it1 in range(Ntau):
            t1 = -1j * it1 * dtau
            for it2 in range(Nt):
                t2 = it2 * dt

                G = -UksI[index,it1]*fks[index]*np.exp(beta*eks[index])
                G = np.einsum('ij,jk,kl,lm->im', R, np.diag(G), np.conj(R).T, np.conj(UksR[index,it2]).T)  
                #G = np.einsum('ij,jk,kl->il',  , G0G, np.conj(UksR[index,it2]).T)
                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.IR[a*Ntau+it1,b*Nt+it2] = G[a,b]


        for it1 in range(Ntau):
            t1 = -1j * it1 * dtau
            for it2 in range(Ntau):
                t2 = -1j * it2 * dtau

                if it1==it2:
                    e1 = (fks[index,0]-0.5) * np.exp(-1j*(+eks[index,0])*(-1j*dtau*(it1-it2)))
                    e2 = (fks[index,1]-0.5) * np.exp(-1j*(+eks[index,1])*(-1j*dtau*(it1-it2)))
                elif it1>it2:
                    e1 = (-fks[index,0]*np.exp(beta*eks[index,0])) * np.exp(-1j*(+eks[index,0])*(-1j*dtau*(it1-it2)))
                    e2 = (-fks[index,1]*np.exp(beta*eks[index,1])) * np.exp(-1j*(+eks[index,1])*(-1j*dtau*(it1-it2)))
                else:
                    e1 = (fks[index,0]) * np.exp(-1j*(+eks[index,0])*(-1j*dtau*(it1-it2)))
                    e2 = (fks[index,1]) * np.exp(-1j*(+eks[index,1])*(-1j*dtau*(it1-it2)))


                G = 1j*np.einsum('ij,j,jk->ik', R, [e1, e2], np.conj(R).T)

                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.M[a*Ntau+it1,b*Ntau+it2] = G[a,b]
                
    return G0
                
def compute_G0_ARPES(ik, myrank, Nk, kpp, k2p, k2i, Nt, Ntau, dt, dtau, fks, UksR, UksI, eks, Norbs):
    G0 = langreth(Nt, Ntau, Norbs)
    
    beta  = dtau*Ntau
    theta = init_theta(Ntau)
        
    kx, ky = get_kx_ky_ARPES(ik, Nk)
    
    # check if this is the right k point for this proc
    # this should have been checked before calling this function
    if myrank==k2p[ik,0]:
        index = k2i[ik,0]

        _, R  = np.linalg.eig(Hk(kx, ky))
        G0L = 1j*np.einsum('ij,j,jk->ik', R, fks[index]-0.0, np.conj(R).T) # - 0.0 for lesser Green's function    
        G0G = 1j*np.einsum('ij,j,jk->ik', R, -fks[index]*np.exp(beta*eks[index]), np.conj(R).T) # - 1.0 for greater Green's function    
        for it1 in range(Nt):
            for it2 in range(Nt):
                G = np.einsum('ij,jk,kl->il', UksR[index,it1], G0L, np.conj(UksR[index,it2]).T)
                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.L[a*Nt+it1,b*Nt+it2] = G[a,b]

                G = np.einsum('ij,jk,kl->il', UksR[index,it1], G0G, np.conj(UksR[index,it2]).T)
                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.G[a*Nt+it1,b*Nt+it2] = G[a,b]


        for it1 in range(Nt):
            t1 = it1 * dt
            for it2 in range(Ntau):
                t2 = -1j * it2 * dtau

                UksI_inv = [UksI[index,it2,1], UksI[index,it2,0]]
                G = fks[index] * UksI_inv
                G = np.einsum('ij,jk,kl,lm->im', UksR[index,it1], R, np.diag(G), np.conj(R).T)
                #G = np.einsum('ij,jk,kl->il', UksR[index,it1], G0L,  )
                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.RI[a*Nt+it1,b*Ntau+it2] = G[a,b]

        for it1 in range(Ntau):
            t1 = -1j * it1 * dtau
            for it2 in range(Nt):
                t2 = it2 * dt

                G = -UksI[index,it1]*fks[index]*np.exp(beta*eks[index])
                G = np.einsum('ij,jk,kl,lm->im', R, np.diag(G), np.conj(R).T, np.conj(UksR[index,it2]).T)  
                #G = np.einsum('ij,jk,kl->il',  , G0G, np.conj(UksR[index,it2]).T)
                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.IR[a*Ntau+it1,b*Nt+it2] = G[a,b]


        for it1 in range(Ntau):
            t1 = -1j * it1 * dtau
            for it2 in range(Ntau):
                t2 = -1j * it2 * dtau

                if it1==it2:
                    e1 = (fks[index,0]-0.5) * np.exp(-1j*(+eks[index,0])*(-1j*dtau*(it1-it2)))
                    e2 = (fks[index,1]-0.5) * np.exp(-1j*(+eks[index,1])*(-1j*dtau*(it1-it2)))
                elif it1>it2:
                    e1 = (-fks[index,0]*np.exp(beta*eks[index,0])) * np.exp(-1j*(+eks[index,0])*(-1j*dtau*(it1-it2)))
                    e2 = (-fks[index,1]*np.exp(beta*eks[index,1])) * np.exp(-1j*(+eks[index,1])*(-1j*dtau*(it1-it2)))
                else:
                    e1 = (fks[index,0]) * np.exp(-1j*(+eks[index,0])*(-1j*dtau*(it1-it2)))
                    e2 = (fks[index,1]) * np.exp(-1j*(+eks[index,1])*(-1j*dtau*(it1-it2)))


                G = 1j*np.einsum('ij,j,jk->ik', R, [e1, e2], np.conj(R).T)

                for a in range(Norbs):
                    for b in range(Norbs):
                        G0.M[a*Ntau+it1,b*Ntau+it2] = G[a,b]
                
    return G0


# do this in the orbital basis
# so D has zeros in off-diagonal blocks
# no U transformations needed
def init_D(omega, Nt, Ntau, dt, dtau, Norbs):

    D = langreth(Nt, Ntau, Norbs)

    beta = dtau*Ntau
    nB   = 1./(np.exp(beta*omega)-1.0)
    theta = init_theta(Ntau)
    theta_transpose = np.transpose(theta)

    for it1 in range(Nt):
        t1 = it1 * dt
        for it2 in range(Nt):
            t2 = it2 * dt
            for ib in range(Norbs):
                D.L[ib*Nt+it1, ib*Nt+it2] = -1j*(nB + 1.0 - 0.0)*np.exp(1j*omega*(t1-t2)) - 1j*(nB + 0.0)*np.exp(-1j*omega*(t1-t2))
                D.G[ib*Nt+it1, ib*Nt+it2] = -1j*(nB + 1.0 - 1.0)*np.exp(1j*omega*(t1-t2)) - 1j*(nB + 1.0)*np.exp(-1j*omega*(t1-t2))


    for it1 in range(Ntau):
        t1 = -1j * it1 * dtau
        for it2 in range(Ntau):
            t2 = -1j * it2 * dtau
            for ib in range(Norbs):
                D.M[ib*Ntau+it1, ib*Ntau+it2] = -1j*(nB + theta_transpose[it1,it2])*np.exp(1j*omega*(t1-t2)) - 1j*(nB +  theta[it1,it2])*np.exp(-1j*omega*(t1-t2))



    for it1 in range(Nt):
        t1 = it1 * dt
        for it2 in range(Ntau):
            t2 = -1j * it2 * dtau
            for ib in range(Norbs):
                D.RI[ib*Nt+it1,ib*Ntau+it2] = -1j*(nB + 1.0 - 0.0)*np.exp(1j*omega*(t1-t2)) - 1j*(nB + 0.0)*np.exp(-1j*omega*(t1-t2))


    for it1 in range(Ntau):
        t1 = -1j * it1 * dtau
        for it2 in range(Nt):
            t2 = it2 * dt
            for ib in range(Norbs):
                D.IR[ib*Ntau+it1,ib*Nt+it2] = -1j*(nB + 1.0 - 1.0)*np.exp(1j*omega*(t1-t2)) - 1j*(nB + 1.0)*np.exp(-1j*omega*(t1-t2))


    return D


def initRA(R, A, L, Nt, Norbs):
    # theta for band case
    theta = init_block_theta(Nt, Norbs)
        
    R =  theta * (L.G - L.L)
    A = -np.transpose(theta) * (L.G - L.L)


def initR(R, theta, L, Nt, Norbs):
    # theta for band case
    #theta = init_block_theta(Nt, Norbs)   
    R =  theta * (L.G - L.L)    
    #R += L.G
    #R -= L.L
    #R *= theta
    
def initA(A, theta, L, Nt, Norbs):
    # theta for band case
    #theta = init_block_theta(Nt, Norbs)
    A = -np.transpose(theta) * (L.G - L.L)
    #A += L.G
    #A -= L.L
    #A *= -np.transpose(theta)
    
        
def computeRelativeDifference(a, b):
    '''
    change = np.sum(abs(a.L - b.L))/np.sum(abs(a.L)) \
           + np.sum(abs(a.G - b.G))/np.sum(abs(a.G)) \
           + np.sum(abs(a.R - b.R))/np.sum(abs(a.R)) \
           + np.sum(abs(a.A - b.A))/np.sum(abs(a.A)) \
           + np.sum(abs(a.RI - b.RI))/np.sum(abs(a.RI)) \
           + np.sum(abs(a.IR - b.IR))/np.sum(abs(a.IR)) \
           + np.sum(abs(a.M - b.M))/np.sum(abs(a.M))
    '''
    
    change = [np.sum(abs(a.L - b.L))/np.sum(abs(a.L)),
              np.sum(abs(a.G - b.G))/np.sum(abs(a.G)),
              np.sum(abs(a.R - b.R))/np.sum(abs(a.R)),
              np.sum(abs(a.A - b.A))/np.sum(abs(a.A)),
              np.sum(abs(a.RI - b.RI))/np.sum(abs(a.RI)),
              np.sum(abs(a.IR - b.IR))/np.sum(abs(a.IR)),
              np.sum(abs(a.M - b.M))/np.sum(abs(a.M))]
   
    return change;
        

def mymult(a, b, c ):
    np.dot(a, b, c);


def multiply(a, b, c, temp_mats, Nt, Ntau, dt, dtau, Norbs):

    aR = temp_mats[0]
    #aA = temp_mats[1]
    #bR = temp_mats[2]
    bA = temp_mats[1]
    #cR = temp_mats[4]
    #cA = temp_mats[5]
    mixed_product = temp_mats[2]
    temp1 = temp_mats[3]
    temp2 = temp_mats[4]
    temp3 = temp_mats[5]
    block_theta = temp_mats[6]
        
    initR(aR, block_theta, a, Nt, Norbs)
    initA(bA, block_theta, b, Nt, Norbs)
    
    '''
    aR  += np.diag(a.DR)
    a.M += np.diag(a.DM)

    bA  += np.diag(b.DR)
    b.M += np.diag(b.DM)
    '''

    for i in range(Nt):
        aR[i,i] += a.DR[i]
        bA[i,i] += b.DR[i]

    for i in range(Ntau):
        a.M[i,i] += a.DM[i]
        b.M[i,i] += b.DM[i]

    
    c.zero(Nt, Ntau, Norbs)

    np.dot(a.M*(-1j*dtau), b.M, c.M)
    
    np.dot(a.RI*(-1j*dtau), b.IR, mixed_product)

    
    np.dot(a.G, bA, temp1)
    c.G  = temp1.copy()
    np.dot(aR, b.G, temp1)
    c.G += temp1
    c.G *= dt
    c.G += mixed_product
    '''

    np.dot(a.L, bA, temp1)
    c.L  = temp1.copy()
    np.dot(aR, b.L, temp1)
    c.L += temp1
    c.L *= dt
    c.L += mixed_product

    np.dot(aR * (dt), b.RI, temp2)
    c.RI = temp2.copy()
    np.dot(a.RI * (-1j*dtau), b.M, temp2)
    c.RI += temp2

    np.dot(a.IR * (dt), bA, temp3)
    c.IR = temp3.copy()
    np.dot(a.M * (-1j*dtau), b.IR, temp3)
    c.IR += temp3
    
    '''

    '''
    #c = langreth(Nt, Ntau, Norbs)
    c.zero(Nt, Ntau, Norbs)
    
    c.M = np.dot(a.M, b.M) * (-1j*dtau)
    
    cR = np.dot(aR, bR) * (dt)
    cA = np.dot(aA, bA) * (dt)

    mixed_product = np.dot(a.RI, b.IR) * (-1j*dtau)
    
    c.G = (np.dot(a.G, bA) + np.dot(aR, b.G)) * (dt) + mixed_product
    c.L = (np.dot(a.L, bA) + np.dot(aR, b.L)) * (dt) + mixed_product
    
    c.RI = np.dot(aR, b.RI) * (dt) + np.dot(a.RI, b.M) * (-1j*dtau)
    c.IR = np.dot(a.IR, bA) * (dt) + np.dot(a.M, b.IR) * (-1j*dtau)
    
    #return c
    '''

#invert a * b = c to solve for b
def solve(a, b, c, temp_mats, Nt, Ntau, dt, dtau, Norbs):

    aR = temp_mats[0]
    aA = temp_mats[1]
    bR = temp_mats[2]
    bA = temp_mats[3]
    cR = temp_mats[4]
    cA = temp_mats[5]
    mixed_product = temp_mats[6]
    
    initRA(aR, aA, a, Nt, Norbs)
    initRA(cR, cA, c, Nt, Norbs)
    
    aR  += np.diag(a.DR)
    aA  += np.diag(a.DR)
    a.M += np.diag(a.DM)

    cR  += np.diag(c.DR)
    cA  += np.diag(c.DR)
    c.M += np.diag(c.DM)
    
    #b = langreth(Nt, Ntau, Norbs)
    b.zero(Nt, Ntau, Norbs)

    
    aMinv = np.linalg.inv(a.M)
    aRinv = np.linalg.inv(aR)
    aAinv = np.linalg.inv(aA)
    
    #b.M = np.dot(aMinv, c.M) / (-1j*dtau)
    np.dot(aMinv, c.M / (-1j*dtau), b.M)
    
    np.dot(aRinv, cR / (dt), bR)
    np.dot(aAinv, cA / (dt), bA)

    temp = np.zeros([Norbs*Nt, Norbs*Ntau], dtype=complex)
    np.dot(a.RI, b.M*(-1j*dtau), temp)
    np.dot(aRinv / (dt), c.RI - temp, b.RI)
    temp = np.zeros([Norbs*Ntau, Norbs*Nt], dtype=complex)
    np.dot(a.IR, bA*(dt), temp)
    np.dot(aMinv / (-1j*dtau), c.IR - temp, b.IR)

    np.dot(a.RI*(-1j*dtau), b.IR, mixed_product)

    temp = np.zeros([Norbs*Nt, Norbs*Nt], dtype=complex)
    np.dot(a.G*(dt), bA, temp)
    np.dot(aRinv / (dt), c.G  - temp  - mixed_product, b.G)
    temp = np.zeros([Norbs*Nt, Norbs*Nt], dtype=complex)
    np.dot(a.L*(dt), bA, temp)
    np.dot(aRinv / (dt), c.L  - temp  - mixed_product, b.L )
    


