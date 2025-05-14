
usage: RCmaxtrain [-h] [--maxpool1 MAXPOOL1] [--dropout1 DROPOUT1]
                  [--convfilt2 CONVFILT2] [--convkern2 CONVKERN2]
                  [--maxpool2 MAXPOOL2] [--dropout2 DROPOUT2]
                  [--convfilt3 CONVFILT3] [--convkern3 CONVKERN3]
                  [--maxpool3 MAXPOOL3] [--dropout3 DROPOUT3]
                  [--dense1 DENSE1] [--lambda1 LAMBDA1] [--lambda2 LAMBDA2]
                  [--rseed RSEED]
                  posfile negfile outprefix

usage: RCmaxpredict [-h] [--maxpool1 MAXPOOL1] [--dropout1 DROPOUT1]
                    [--convfilt2 CONVFILT2] [--convkern2 CONVKERN2]
                    [--maxpool2 MAXPOOL2] [--dropout2 DROPOUT2]
                    [--convfilt3 CONVFILT3] [--convkern3 CONVKERN3]
                    [--maxpool3 MAXPOOL3] [--dropout3 DROPOUT3]
                    [--dense1 DENSE1] [--lambda1 LAMBDA1] [--lambda2 LAMBDA2]
                    [--rseed RSEED]
                    seqfile outprefix

installation:  (this works for anaconda python3, sh Anaconda3-2020.02-Linux-x86_64.sh)
python --version
3.7.6
git clone git@github.com:mbeer3/RCmax.git
python -m venv venv
source venv/bin/activate
pip install bitarray==1.9.0
pip install sklearn==0.0
pip install matplotlib==3.3.3
pip install keras==2.5.0rc0
pip install tensorflow==2.5.0
pip install tensorflow-gpu==2.5.0
pip install theano==1.0.5

example:

mkdir runs

./RCmaxtrain --rseed 8 DHS_E2_72_300_noproms_nc30_hg38_top10k.fa neg1x_r2_DHS_E2_72_300_noproms_nc30_hg38_top10k.fa DHS_E2_72_300_noproms_nc30_hg38_top10k_vs_neg1x_r2_8

./RCmaxpredict DHS_E2_72_300_noproms_nc30_hg38_top10k.fa DHS_E2_72_300_noproms_nc30_hg38_top10k_vs_neg1x_r2_8

./RCmaxtrain --rseed 8 DHS_553_hg38_300.fa neg1x_r1_DHS_553_hg38_300.fa DHS_553_hg38_300_vs_neg1x_r1_8

./RCmaxpredict DHS_553_hg38_300.fa DHS_553_hg38_300_vs_neg1x_r1_8

# RCmax
