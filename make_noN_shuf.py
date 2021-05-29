import numpy, scipy, random, scipy.io, h5py, os, sys

def shuffle(a):
    out=[]
    while len(a)>0:
        i=random.randrange(len(a))
        out.append(a[i])
        a=a[0:i]+a[i+1:]
    return out

def main():
    random.seed(eval(sys.argv[2]))
    infile=open(sys.argv[1],'r')
    ofile=open(sys.argv[1][:-3]+'_noN.fa','w')
    lines=infile.readlines()
    infile.close()
    index=[]
    labels=[]
    seq={}
    for i in range(len(lines)):
        line=lines[i]
        if line[0:1]==">":
            seq[line]=lines[i+1].strip()
            labels.append(line)
#    print(shuffle(['d','d1','d2','a','dog']))
#    print(shuffle(['d','d1','d2','a','dog']))

    index=shuffle(labels)
    
    for x in index:
        ofile.write(x)
        seq0=seq[x]
        seq2=''
        for i in range(len(seq0)):
            seq1=seq0[i]
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
            seq2=seq2+seq1
        ofile.write(seq2+'\n')
    ofile.close()

main()

    
