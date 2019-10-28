import sys,os
import numpy as np
from glob import glob

def get_counts(dirpath):

    #myfiles = glob(dirpath+'ARPES'+LG+'/*')
    myfiles = glob(dirpath+'*')

    counts = {}

    for myfile in myfiles:
        i1 = myfile.find('I')
        i2 = myfile.find('_', i1)
        i3 = myfile.find('.', i1)
        ik = int(myfile[i1+1:i2])
        it = int(myfile[i2+1:i3])

        if ik not in counts:
            counts[ik] = it
        elif ik in counts and counts[ik]<it:
            counts[ik] = it

    #print 'counts'
    #print counts

    return counts

def main(dirpath, Nk, Nw):

    #myfiles = glob(dirpath+'ARPES'+LG+'/*')
    myfiles = glob(dirpath+'*')

    print 'len files',len(myfiles)

    assert len(myfiles)%Nk==0

    Nt = len(myfiles)//Nk

    I = np.zeros([Nk, Nw, Nt])

    for i,myfile in enumerate(myfiles):
        if i%(len(myfiles)//100)==0:
            print '%1.2f'%(1.*i/len(myfiles))

        i1 = myfile.find('I')
        i2 = myfile.find('_', i1)
        i3 = myfile.find('.', i1)
        ik = int(myfile[i1+1:i2])
        it = int(myfile[i2+1:i3])
        
        #print ik,it
        I[ik,:,it] = np.load(myfile)

    np.save('I.npy', I)
    

if __name__=='__main__':
    dirpath = sys.argv[1]
    #LG = sys.argv[2]
    Nk = 96
    Nw = 100

    counts = get_counts(dirpath)

    print counts

    #exit()
    if(raw_input("continue?")!='y'):
        exit()

    mymin = None
    mymax = None

    for ik,v in counts.iteritems():
        if mymin==None:
            mymin = v
            mymax = v
        else:
            if v<mymin:
                mymin = v
            if v>mymax:
                mymax = v

    print 'min, max',mymin,mymax
    assert mymin==mymax

    main(dirpath, Nk, Nw)


