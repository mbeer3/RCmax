import numpy, scipy, random, scipy.io, h5py, os, sys, os.path

def make_data(pfile):

    seq=[]
    class1=[]
    used=[]
    name=[]
    plines=pfile.readlines()
    ntot=int((len(plines))/2)
    seqlen=len(plines[1].strip())
    print('seqlen:',seqlen)
    seq4=numpy.zeros((ntot,4,seqlen),numpy.bool)

    for i in range(len(plines)):
        if i<len(plines):
            line=plines[i]
            if line[0:1]==">":
                name.append(line[1:-1])
            else:
                seq.append(line.strip())
                class1.append(1)
                used.append(0)
    print(class1[0:20])
    print(len(used),'seqs')

    seql=[]
    for i in range(len(used)):
        if i%ntot/2==0:
            print(i)
        seq2=seq[i]
        #print len(seq2)
        for j in range(len(seq2)):
            seq1=seq2[j]
            if seq1=="N":
                if random.randrange(2)==0:
                    if random.randrange(2)==0:
                        seq1="A"
                    else:
                        seq1="G"
                else:
                    if random.randrange(2)==0:
                        seq1="C"
                    else:
                        seq1="T"
            if seq1=="A":
                seq4[i,0,j]=1
            elif seq1=="C":
                seq4[i,1,j]=1
            elif seq1=="G":
                seq4[i,2,j]=1
            elif seq1=="T":
                seq4[i,3,j]=1
            else:
                print("error",seq1)
    return seq4, class1, name


def make_file(seq4, class1, seq4rc, class1_rc, outpre,name):

    print('seq4 size:',seq4.shape)
    (ntot,b,seqlen)=seq4.shape
    ntest=0
    nvalid=0
    print('seq4 size:',seq4.shape,ntot,seqlen,ntest,nvalid)
#    n=63339-5700-5700
    n=ntot-ntest-nvalid
   #  n=22384-1000-4000
    print(n,'training seqs')
#    valid=numpy.zeros((nvalid,1),numpy.bool)
#    validx=numpy.zeros((nvalid,4,2*seqlen),numpy.bool)
#    test1=numpy.zeros((ntest,1),numpy.bool)
#    test1x=numpy.zeros((ntest,4,2*seqlen),numpy.bool)
    train1=numpy.zeros((n,1),numpy.bool)
    train1x=numpy.zeros((n,4,2*seqlen),numpy.bool)
#    namex=numpy.empty((n,1),dtype='S')
#    namex=numpy.array((n,1),numpy.str_)
    dt = h5py.special_dtype(vlen=str)
    namex=numpy.array(name,dtype=dt)
    
    counter =0
    for l in range(ntot):
        if class1[l]==class1_rc[l]:
            counter+=1

    print('value of counter is = ',counter)
    print('valid:')
    k=0
    while k<nvalid:
        valid[k,0]=class1[k]
        for i1 in range(4):
            for i2 in range(seqlen):
                validx[k,i1,i2]=seq4[k,i1,i2]
            for i3 in range(seqlen):
                validx[k,i1,i3+seqlen]=seq4rc[k,i1,i3]
        k=k+1

    print('test:')
            
    k=nvalid
    while k<nvalid+ntest:
        test1[k-nvalid,0]=class1[k]
        for i1 in range(4):
            for i2 in range(seqlen):
                test1x[k-nvalid,i1,i2]=seq4[k,i1,i2]
            for i3 in range(seqlen):
                test1x[k-nvalid,i1,i3+seqlen]=seq4rc[k,i1,i3]
        k=k+1


    print('train:')

    k=nvalid+ntest
    while k<ntot:
        train1[k-nvalid-ntest,0]=class1[k]
#        namex[k,0]=name[k]
#        namex[k,0]='bob'
        for i1 in range(4):
            for i2 in range(seqlen):
                train1x[k-nvalid-ntest,i1,i2]=seq4[k,i1,i2]
            for i3 in range(seqlen):
                train1x[k-nvalid-ntest,i1,i3+seqlen]=seq4rc[k,i1,i3]
        k=k+1

    print('saving the 500 bp sequence')    

#    scipy.io.savemat('test1b.mat',{'testdata':test1,'testxdata':test1x})
    print('saving *.h5')    
    print(train1.shape)
    print(train1x.shape)
#    scipy.io.savemat('train1b.mat',{'traindata':train1,'trainxdata':train1x})
    v=h5py.File(outpre + '.h5','w')
    v.create_dataset('traindata',data=train1)
    v.create_dataset('trainxdata',data=train1x)
    v.create_dataset('namedata',data=namex)
    v.close()


def main(argv = sys.argv):
    posfa = argv[1]
    outpre = argv[2]
    if not os.path.isfile(argv[1][0:-3]+'_noN.fa'):
        print('making noN sequence')
        os.system('python make_noN.py {0} {1} '.format(posfa))
    if not os.path.isfile(argv[1][0:-3]+'_noN_rc.fa'):
        print('making revcomp sequence')
        os.system('python make_rc_fasta.py {0}_noN.fa'.format(posfa[:-3]))
    posfan = argv[1][0:-3]+'_noN.fa'
    posfa_rc = argv[1][0:-3]+'_noN_rc.fa'
    pfile=open(posfan,'r')

    seq, class1, name=make_data(pfile)

    pfile_rc=open(posfa_rc,'r')

    seq_rc, class1_rc, name1=make_data(pfile_rc)

    make_file(seq, class1, seq_rc, class1_rc,outpre,name)

main()

    
