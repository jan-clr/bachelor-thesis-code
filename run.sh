# python train.py -bs 10 -rn res34d_upConv_AlbuV1_s320 -enc resnet34d -lr 1e-4 -lrsp 100 -lrsf 0.1 -mf "../runs/Cityscapes/res34d_upConv_AlbuV1_lrsp_5_lrsf_0.5_bs_16_lr_0.0001/model_best.pth.tar";
# python train.py -bs 16 -rn res34d_nomt_aug2_100lbls -enc resnet34d -lr 1e-4 --no-lrs -lrsp 15 -lrsf 0.5 -lblr -100 --no-mt;
python train.py -bs 16 -rn res34d_mt_aug2_lrson_100lbls -enc resnet34d -lr 1e-4 --lrs -lrsp 15 -lrsf 0.5 -lblr -100 -ulblr 100- --dropout 0.5 --mtdelay 10;
python train.py -bs 16 -rn res34d_mt_aug2_lrson_100lbls -enc resnet34d -lr 1e-3 --lrs -lrsp 15 -lrsf 0.5 -lblr -100 -ulblr 100- --dropout 0.5 --mtdelay 10;
