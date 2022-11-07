# python train.py -bs 10 -rn res34d_upConv_AlbuV1_s320 -enc resnet34d -lr 1e-4 -lrsp 100 -lrsf 0.1 -mf "../runs/Cityscapes/res34d_upConv_AlbuV1_lrsp_5_lrsf_0.5_bs_16_lr_0.0001/model_best.pth.tar";
python train.py -bs 16 -rn res34d_upConv_AlbuV2_fullres -enc resnet34d -lr 1e-4 -lrsp 10 -lrsf 0.5;
