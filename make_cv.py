import numpy, scipy, random, scipy.io, h5py, os, sys, os.path

def make_data(pfile, nfile):

    seq=[]
    class1=[]
    used=[]
    plines=pfile.readlines()
    nlines=nfile.readlines()
    ntot=int((len(plines)+len(nlines))/2)
    seqlen=len(plines[1].strip())
    print('seqlen:',seqlen)
    seq4=numpy.zeros((ntot,4,seqlen),numpy.bool)

    half=int(len(nlines)/2)

    for i in range(max(len(plines),len(nlines))):
        if i<len(plines):
            line=plines[i]
            if line[0:1]==">":
                pass
            else:
                seq.append(line.strip())
                class1.append(1)
                used.append(0)
        if i<len(nlines):
            line=nlines[i]
            if line[0:1]==">":
                pass
            else:
                seq.append(line.strip())
                class1.append(0)
                used.append(0)
    print(class1[0:20])
    print(len(used),'combined pos and neg seqs')

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
    return seq4, class1


def make_file(seq4, class1, seq4rc, class1_rc, outpre):

    print('seq4 size:',seq4.shape)
    (ntot,b,seqlen)=seq4.shape
    ntest=int(ntot*.1)
    nvalid=int(ntot*.1)
    print('seq4 size:',seq4.shape,ntot,seqlen,ntest,nvalid)
#    n=63339-5700-5700
    n=ntot-ntest-nvalid
   #  n=22384-1000-4000
    print(n,'training seqs')
    valid=numpy.zeros((nvalid,1),numpy.bool)
    validx=numpy.zeros((nvalid,4,2*seqlen),numpy.bool)
    test1=numpy.zeros((ntest,1),numpy.bool)
    test1x=numpy.zeros((ntest,4,2*seqlen),numpy.bool)
    train1=numpy.zeros((n,1),numpy.bool)
    train1x=numpy.zeros((n,4,2*seqlen),numpy.bool)

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
        for i1 in range(4):
            for i2 in range(seqlen):
                train1x[k-nvalid-ntest,i1,i2]=seq4[k,i1,i2]
            for i3 in range(seqlen):
                train1x[k-nvalid-ntest,i1,i3+seqlen]=seq4rc[k,i1,i3]
        k=k+1

    print('saving the 500 bp sequence')    

    print(valid[0:10])
    print('saving valid_seed.h5')
    print(valid.shape)
    print(validx.shape) 
    v=h5py.File(outpre+ '_valid.h5','w')
    v.create_dataset('validdata',data=valid)
    v.create_dataset('validxdata',data=validx)
    v.close()
    print('saving test_seed.h5')
    print(test1.shape)
    print(test1x.shape)
    v=h5py.File(outpre + '_test.h5','w')
    v.create_dataset('testdata',data=test1)
    v.create_dataset('testxdata',data=test1x)
    v.close()
#    scipy.io.savemat('test1b.mat',{'testdata':test1,'testxdata':test1x})
    print('saving train_seed.h5')    
    print(train1.shape)
    print(train1x.shape)
#    scipy.io.savemat('train1b.mat',{'traindata':train1,'trainxdata':train1x})
    v=h5py.File(outpre + '_train.h5','w')
    v.create_dataset('traindata',data=train1)
    v.create_dataset('trainxdata',data=train1x)
    v.close()


def main(argv = sys.argv):
    posfa = argv[1]
    negfa = argv[2]
    outpre = argv[3]
    rseed = argv[4]
    if not os.path.isfile(argv[1][0:-3]+'_noN.fa'):
        print('making noN shuffled sequence')
        os.system('python make_noN_shuf.py {0} {1} '.format(posfa,rseed))
    if not os.path.isfile(argv[2][0:-3]+'_noN.fa'):
        print('making noN shuffled sequence')
        os.system('python make_noN_shuf.py {0} {1} '.format(negfa,rseed))
    if not os.path.isfile(argv[1][0:-3]+'_noN_rc.fa'):
        print('making revcomp sequence')
        os.system('python make_rc_fasta.py {0}_noN.fa'.format(posfa[:-3]))
    if not os.path.isfile(argv[2][0:-3]+'_noN_rc.fa'):
        print('making revcomp sequence')
        os.system('python make_rc_fasta.py {0}_noN.fa'.format(negfa[:-3]))
    posfan = argv[1][0:-3]+'_noN.fa'
    negfan = argv[2][0:-3]+'_noN.fa'
    posfa_rc = argv[1][0:-3]+'_noN_rc.fa'
    negfa_rc = argv[2][0:-3]+'_noN_rc.fa'
    pfile=open(posfan,'r')
    nfile=open(negfan,'r')

    seq, class1=make_data(pfile,nfile)

    pfile_rc=open(posfa_rc,'r')
    nfile_rc=open(negfa_rc,'r')

    seq_rc, class1_rc=make_data(pfile_rc,nfile_rc)

    make_file(seq, class1, seq_rc, class1_rc,outpre)

main()

    
