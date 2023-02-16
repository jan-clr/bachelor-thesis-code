# Detection and Measurement of SAW Vaporized Fluids using ML

The code for my bachelor thesis regarding detection and measurement of vaporized droplets using ML Models. 

## Installation

After cloning this repository I recommend creating a virtuel environment by running 
```bash
python -m venv .venv
```
and activating it in the way that is appropriate for your system. 

Running
```bash
pip install .
```
will install all necessary dependencies for the application part of the project, however if you want to train your own models using this code as a basis you will need to install additional dependencies, as well as initialize the git submodules.

## Usage

### Training

The script to execute for model training is `train.py`. It takes a number of arguments, which can be seen by running `python train.py --help`, which will generate the following output:
```bash
usage: train.py [-h] [-rn RUNNAME] [--model MODEL] [--optimizer OPTIMIZER] [-enc ENCODER] [-lr LR] [-bs BS] [-ds DS] [-bsul BSUL] [-lrsp LRSP] [-lrsf LRSF] [-mf MF] [-lblr LABELEDRANGE]
                [-ulblr UNLABELEDRANGE] [--lrs | --no-lrs] [--mt | --no-mt] [--ctn | --no-ctn] [--dropout DROPOUT] [--dropout_tch DROPOUT_TCH] [--mtdelay MTDELAY] [--iter | --no-iter]
                [--skip | --no-skip] [--split SPLIT] [--oidx [OIDX ...]]

Start Model Training.

options:
  -h, --help            show this help message and exit
  -rn RUNNAME, --runname RUNNAME
                        Use a specific name for the run
  --model MODEL         Model to use for training. Should be either 'unet' or 'dlv3p'
  --optimizer OPTIMIZER
                        Optimizer to use for training. Should be either 'adam' or 'sgd'
  -enc ENCODER, --encoder ENCODER
                        Name of the timm model to use as the encoder
  -lr LR                Set the initial learning rate
  -bs BS                Set the batch size
  -ds DS                Set the dataset name
  -bsul BSUL            Set the unlabeled batch size
  -lrsp LRSP            Set the Patience for the learning rate scheduler
  -lrsf LRSF            Set the Factor used to reduce the learning rate
  -mf MF                Load from non default file
  -lblr LABELEDRANGE, --labeledrange LABELEDRANGE
                        Use this range of the dataset as labeled samples.
  -ulblr UNLABELEDRANGE, --unlabeledrange UNLABELEDRANGE
                        Use this range of the dataset as unlabeled samples.
  --lrs, --no-lrs       Set if learning rate scheduler should be used
  --mt, --no-mt         Set if mean teacher should be used
  --ctn, --no-ctn       Continue previous run
  --dropout DROPOUT     Set model dropout rate
  --dropout_tch DROPOUT_TCH
                        Set teacher dropout rate when using MT
  --mtdelay MTDELAY     Set a number of epochs to train only on labeled data before mean teacher sets in
  --iter, --no-iter     Use iterative semi supervised learning approach.
  --skip, --no-skip     Skip supervised training in iterative approach when loading a model.
  --split SPLIT         Set the factor for splitting the training images
  --oidx [OIDX ...]     Set the out indices for the feature extractor.

```

Most of the arguments are self explanatory, with the only non canonical formats being the `--labeledrange` and `--unlabeledrange` arguments. These arguments take a string of the format `start-end` where `start` is the index of the first sample to use, `end` is the index of the last sample to use (exclusive), with either being optional. For example, `--labeledrange 0-100` will use the first 100 samples of the dataset as labeled samples, while `--unlabeledrange 100-` will use every sample from 100th sample as unlabeled samples. 
Omitting either start or end is equivalent to inputting `None` into the python `slice(start, end)` syntax.

`split` is the factor used to split the training images into patches. For example, if `split` is set to 2, the image will be split into 4 patches, each of which will be used as a training sample.

`oidx` is a list of space separated indices of the output of the feature extractor to use. For example, if `oidx` receives `0 1 2`, only the first three layers of the feature extractor will be used, if applicable.

#### Datasets
By default, datasets are expected to be placed in the `data` folder inside the top level of the repository; to specify another location, change the `ROOT_DATA_DIR` constant in the `train.py` file.

Datasets are expected to be in the Cityscapes format, with the following structure:
```
dataset_name
├── gtFine
│   ├── train
│   │   ├── subset1
│   │   │   ├── 0_gtFine_color.png
│   │   │   ├── 0_gtFine_instanceIds.png
│   │   │   ├── 0_gtFine_labelIds.png
│   │   │   └── ...
│   │   ├── subset2
│   │   └── ...
│   └── val
│       ├── subsetx
│       └── ...
├── leftImg8bit
│   ├── train
│   │   ├── subset1
│   │   │   ├── 0_letImg8bit.png
│   │   │   └── ...
│   │   ├── subset2
│   │   └── ...
│   └── val
│       ├── subsetx
│       └── ...
├── gtFine.zip
└── leftImg8bit.zip
```

If you place the zip archives in the dataset folder, they will be automatically extracted.

#### Models 
For training, model architecture needs to be defined in the code (not for inference). The models used by this project can be found in the `models.py` file.
Instantiating a model for training is done in the `create_models` method in the train script. If you want to use a different model, you will need to add it to this method.

