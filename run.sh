# python train.py -bs 10 -rn res34d_upConv_AlbuV1_s320 -enc resnet34d -lr 1e-4 -lrsp 100 -lrsf 0.1 -mf "../runs/Cityscapes/res34d_upConv_AlbuV1_lrsp_5_lrsf_0.5_bs_16_lr_0.0001/model_best.pth.tar";
# python train.py -bs 16 -rn res34d_mt_aug1 -enc resnet34d -lr 1e-4 -lrsp 10 -lrsf 0.5 -lblr -1000 -ulblr 1000-;
python train.py -bs 16 -rn res34d_mt_aug1_lrsp_10_lrsf_0.5_bs_16_lr_0.0001_p_40_24-11-2022_12-59-40 -enc resnet34d -lr 1e-4 --no-lrs -lrsp 10 -lrsf 0.5 -lblr -1000 -ulblr 1000-;
