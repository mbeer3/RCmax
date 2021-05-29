
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

example:

mkdir runs

./RCmaxtrain --rseed 8 DHS_E2_72_300_noproms_nc30_hg38_top10k.fa neg1x_r2_DHS_E2_72_300_noproms_nc30_hg38_top10k.fa DHS_E2_72_300_noproms_nc30_hg38_top10k_vs_neg1x_r2_8

./RCmaxpredict DHS_E2_72_300_noproms_nc30_hg38_top10k.fa DHS_E2_72_300_noproms_nc30_hg38_top10k_vs_neg1x_r2_8

# RCmax
