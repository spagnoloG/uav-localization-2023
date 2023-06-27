# CustomResNetDeiT for Multi-View Image Fusion
This repository contains code for a custom implementation of a Vision Transformer model which uses a pretrained ResNet50 for feature extraction 
and a Transformer for fusing features from UAV and satellite view images. The model uses a novel loss function called Balance Loss, 
designed to handle imbalanced datasets by adjusting the weights of positive and negative samples.

## Deps 

```bash
pip install -r requirements.txt
```
## Model Architecture


### Twins Backbone
```bash
>>> import torch
>>> from timm import create_model
>>>
>>> model = create_model('twins_pcpvt_small', pretrained=True)
>>> print(model)
Twins(
  (patch_embeds): ModuleList(
    (0): PatchEmbed(
      (proj): Conv2d(3, 64, kernel_size=(4, 4), stride=(4, 4))
      (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    )
    (1): PatchEmbed(
      (proj): Conv2d(64, 128, kernel_size=(2, 2), stride=(2, 2))
      (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    )
    (2): PatchEmbed(
      (proj): Conv2d(128, 320, kernel_size=(2, 2), stride=(2, 2))
      (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
    )
    (3): PatchEmbed(
      (proj): Conv2d(320, 512, kernel_size=(2, 2), stride=(2, 2))
      (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
  )
  (pos_drops): ModuleList(
    (0-3): 4 x Dropout(p=0.0, inplace=False)
  )
  (blocks): ModuleList(
    (0): ModuleList(
      (0-2): 3 x Block(
        (norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (attn): GlobalSubSampleAttn(
          (q): Linear(in_features=64, out_features=64, bias=True)
          (kv): Linear(in_features=64, out_features=128, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=64, out_features=64, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
          (sr): Conv2d(64, 64, kernel_size=(8, 8), stride=(8, 8))
          (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        )
        (drop_path1): Identity()
        (norm2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=64, out_features=512, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=512, out_features=64, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
        (drop_path2): Identity()
      )
    )
    (1): ModuleList(
      (0-3): 4 x Block(
        (norm1): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        (attn): GlobalSubSampleAttn(
          (q): Linear(in_features=128, out_features=128, bias=True)
          (kv): Linear(in_features=128, out_features=256, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=128, out_features=128, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
          (sr): Conv2d(128, 128, kernel_size=(4, 4), stride=(4, 4))
          (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        )
        (drop_path1): Identity()
        (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=128, out_features=1024, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=1024, out_features=128, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
        (drop_path2): Identity()
      )
    )
    (2): ModuleList(
      (0-5): 6 x Block(
        (norm1): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
        (attn): GlobalSubSampleAttn(
          (q): Linear(in_features=320, out_features=320, bias=True)
          (kv): Linear(in_features=320, out_features=640, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=320, out_features=320, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
          (sr): Conv2d(320, 320, kernel_size=(2, 2), stride=(2, 2))
          (norm): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
        )
        (drop_path1): Identity()
        (norm2): LayerNorm((320,), eps=1e-06, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=320, out_features=1280, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=1280, out_features=320, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
        (drop_path2): Identity()
      )
    )
    (3): ModuleList(
      (0-2): 3 x Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): GlobalSubSampleAttn(
          (q): Linear(in_features=512, out_features=512, bias=True)
          (kv): Linear(in_features=512, out_features=1024, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path1): Identity()
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=512, out_features=2048, bias=True)
          (act): GELU(approximate='none')
          (drop1): Dropout(p=0.0, inplace=False)
          (norm): Identity()
          (fc2): Linear(in_features=2048, out_features=512, bias=True)
          (drop2): Dropout(p=0.0, inplace=False)
        )
        (drop_path2): Identity()
      )
    )
  )
  (pos_block): ModuleList(
    (0): PosConv(
      (proj): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
      )
    )
    (1): PosConv(
      (proj): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
      )
    )
    (2): PosConv(
      (proj): Sequential(
        (0): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=320)
      )
    )
    (3): PosConv(
      (proj): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
      )
    )
  )
  (norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
  (head_drop): Dropout(p=0.0, inplace=False)
  (head): Linear(in_features=512, out_features=1000, bias=True)
)
```
