import numpy as np
import h5py
import sys

f = open(sys.argv[1],'r')


d={'A':'T','C':'G','G':'C','T':'A'}


f_rc_on=sys.argv[1][:-3]+'_rc.fa'

outfile=open(f_rc_on,'w')

for line in f:
        x=line.split()
        if x[0].startswith('>'):
                #print ''
                outfile.write(x[0]+'_rc'+'\n')
        else:
                seq=x[0]
                #print seq
                rev=seq[::-1]
                rc_out=''
                for bp in rev:
                        rc_out+=d[bp]
                #print rc_out
                outfile.write(rc_out+'\n')
outfile.close()
