# CustomResNetDeiT for Multi-View Image Fusion
This repository contains code for a custom implementation of a Vision Transformer model which uses a pretrained ResNet50 for feature extraction 
and a Transformer for fusing features from UAV and satellite view images. The model uses a novel loss function called Balance Loss, 
designed to handle imbalanced datasets by adjusting the weights of positive and negative samples.

## Deps 

```bash
pip install -r requirements.txt
```
## Model Architecture
The model architecture comprises of the following components:

- Feature Extractor: A ResNet50 model pretrained on ImageNet is used to extract features from the input images.
- Positional Encoding: A positional encoding layer is used to capture positional information of the patches.
- Image Transformer: A transformer is used to fuse the features from the UAV and satellite view images.
- Balance Loss: A custom loss function is used to balance the positive and negative samples in the dataset.
