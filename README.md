# microUS-seg
Publicly available micro ultrasound segmentation models

## Installation
1. Create a python environment and activate it
2. Install requirements: `torch==2.6.0`, `monai==1.5.0`

## Available models
List available models using `torch.hub.list('pfrwilson/microUS-seg')`.
You can instantiate a model in one line using `torch.hub`, e.g, `model = torch.hub.load('pfrwilson/microUS-seg', model_name)`. 

_input_: Models expect as input image tensors of shape `B x C x H x W`. The orientation of the image should be such that when displayed, the rectal wall is at the bottom of the image and the anterior prostate towards the top of the image. The model has an attribute called `preprocessing_config` specifying the expected input size `H x W`, expected number of input channels `C`, and mean and standard deviation statistics required to correctly normalize the image (starting from an image with pixel values between 0 and 1). 

_output_: Models output are `B x N x H x W`, where `N` is the number of classes (could be 1 for binary segmentation). Outputs are raw logits - use `.sigmoid()` or `.softmax(...)` to convert to probabilities and threshold appropriately to create a binary mask.

