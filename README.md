# Super-Resolution-with-FSRCNN

Tensorflow implementation of FSRCNN actitecture from paper [Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/abs/1608.00367)

### Difference with the Original Paper
- Trained on [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- Added sigmoid activation in the output layer to aling model predictions with groung truth HR images, so their values are in the same range - \[0, 1\]. 
- RMSprop optimizer was used instead of SGD. RMSprop shows faster and more stable training.
- ReduceLROnPlateau and EarlyStopping callbacks addded

### Usage

Train with command: ```python train.py --config config.yaml```

Notebook for inference - Inference.ipynb. 

