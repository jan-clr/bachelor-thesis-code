# Load Best Model and run on vap
# python train.py -bs 10 -rn res34d_upConv_AlbuV1_s320 -enc resnet34d -lr 1e-4 -lrsp 100 -lrsf 0.1 -mf "../runs/Cityscapes/res34d_upConv_AlbuV1_lrsp_5_lrsf_0.5_bs_16_lr_0.0001/model_best.pth.tar";

# Best MT run
# python train.py -bs 16 -rn res34d_nomt_aug2_100lbls_lrsp_30_lrsf_0.5_bs_16_lr_0.0001_p_30_16-12-2022_23-10-57 -enc resnet34d -lr 1e-4 --lrs -lrsp 30 -lrsf 0.5 -lblr -100 --no-mt --ctn;

# Iterative Training
# python train.py -bs 16 -rn res34d_iterative_aug2_100lbls_lrsp_20_lrsf_0.5_bs_16_lr_0.0001_p_50_20-12-2022_22-13-41 -enc resnet34d -lr 1e-4 --lrs -lrsp 15 -lrsf 0.5 -lblr -100 -ulblr 100- --no-mt --iter --ctn --skip;

# Train Deeplab on Cityscapes
#python train.py -bs 8 -bsul 6 -rn nomt_aug2_100lbls --model dlv3p_smp -lr 1e-2 -lrsp 15 -lrsf 0.5 -lblr -100 -ulblr 100- --mt --no-iter;
#python train.py -bs 8 -rn nomt_aug2_alllbls --model dlv3p_smp -lr 1e-2 -lrsp 15 -lrsf 0.5 --no-mt --no-iter;

# Train on vapour dataset 
#python train.py -ds vapourbase -bs 12 -rn nomt_res34 --model unet -enc resnet34d -lr 1e-4 -lrsp 30 -lrsf 0.1 --no-mt --no-iter;
python train.py -ds vapourbase -bs 12 -rn nomt_res34_adam_noweights_noempty --model unet -enc resnet34d -lr 1e-3 -lrsp 15 -lrsf 0.1 --no-mt --no-iter --optimizer adam; # -mf runs/Cityscapes/unet/res34d_fulllbls_normend_lrsp_10_lrsf_0.5_bs_16_lr_0.0001_p_30_21-01-2023_22-17-27/model_best.pth.tar;
python train.py -ds vapourbase -bs 12 -rn nomt_preres34_adam_noweights_noempty --model unet -enc resnet34d -lr 1e-3 -lrsp 15 -lrsf 0.1 --no-mt --no-iter --optimizer adam -mf runs/Cityscapes/unet/res34d_fulllbls_normend_lrsp_10_lrsf_0.5_bs_16_lr_0.0001_p_30_21-01-2023_22-17-27/model_best.pth.tar;
# python train.py -ds vapourbase -bs 8 -bsul 6 -rn mt_res34_sgd --model unet -enc resnet34d -lr 1e-3 -lrsp 30 -lrsf 0.5 --mt -lblr -488 -ulblr 488- --dropout 0.5 --no-iter; # -mf runs/Cityscapes/unet/res34d_fulllbls_normend_lrsp_10_lrsf_0.5_bs_16_lr_0.0001_p_30_21-01-2023_22-17-27/model_best.pth.tar;

# Rerun best Training on Cityscapes
# python train.py -ds Cityscapes --model unet -bs 16 -rn res34d_fulllbls_normend -enc resnet34d -lr 1e-4 -lrsp 10 -lrsf 0.5 --no-mt --no-iter;
