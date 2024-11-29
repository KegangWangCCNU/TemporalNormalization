# A Plug-and-Play Temporal Normalization Module for Robust Remote Photoplethysmography (rPPG)
**The open-source code for [this paper](https://www.arxiv.org/pdf/2411.15283).**

![image](https://github.com/KegangWangCCNU/PICS/blob/main/TNprinciple.jpg)  

## Inserting TN into a 3D convolutional model (Pytorch)
Please refer to [PhysNet_TN_rPPG-Toolbox.py](https://github.com/KegangWangCCNU/TemporalNormalization/blob/main/PhysNet_TN_rPPG-Toolbox.py)
```python
self.ConvBlock1 = nn.Sequential(
    TNM(), # Insert a TN module
    nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
    nn.BatchNorm3d(16),
    nn.ReLU(inplace=True),)

self.ConvBlock2 = nn.Sequential(
    TNM(), # Insert a TN module
    nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
    nn.BatchNorm3d(32),
    nn.ReLU(inplace=True),)
```

## Inserting TN into any end-to-end model (Pytorch)
It is important to note that the TN module must operate on the temporal axis. For instance, in a transformer, it is necessary to first reshape and isolate the temporal axis before applying TN. 
```python
# Reshape to B, T, H, W, C features
x = x.view(-1, t//4, 4, 4, x.size(-1))

# Apply TN on T axis
x = TNM(axis=1)(x)

# Reshape to transformers features
x = x.view(x.size(0), -1, x.size(-1))

x = self.transformer_block(x)
```

For models similar to TS-CAN, in PyTorch, its feature dimension is `(BxT, C, H, W)`. Therefore, it is necessary to specify `axis=0` and `frames=T`. 
```python
# Taking TS-CAN modification in rPPG-Toolbox as an example. 
def forward(self, inputs, params=None):

    # Do not use diff with TN! RAW only!

    # diff_input = inputs[:, :3, :, :]
    raw_input = inputs[:, 3:, :, :]            # (Bx160, C, H, W)

    d0 = TNM(axis=0, frames=160)(raw_input)    # Assuming the chunk size is 160, apply TN
    # diff_input = self.TSM_1(diff_input)
    d0 = self.TSM_1(d0)
    d1 = torch.tanh(self.motion_conv1(d0))

    d1 = TNM(axis=0, frames=160)(d1)           # Apply TN
    d1 = self.TSM_2(d1)
    d2 = torch.tanh(self.motion_conv2(d1))
```

## Inserting TN into any end-to-end model (Keras)
Please refer to the source code [TN/model_tn.py](https://github.com/KegangWangCCNU/TemporalNormalization/blob/main/TN/model_tn.py).

## Implementation ([rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)) 
Please refer to [PhysNet_TN_rPPG-Toolbox.py](https://github.com/KegangWangCCNU/TemporalNormalization/blob/main/PhysNet_TN_rPPG-Toolbox.py) and [Example_rPPG-Toolbox.yaml](https://github.com/KegangWangCCNU/TemporalNormalization/blob/main/Example_rPPG-Toolbox.yaml) in the source code.
`Example_rPPG-Toolbox.yaml` is its configuration file.  

* __Note: For rPPG-Toolbox, do not use any preprocessing or postprocessing, and ensure that both the input and output in the yaml are set to 'Raw'.__

![image](https://github.com/user-attachments/assets/e7603792-48fd-4a8a-9aa7-e7c49e5ddf82)


## Benchmark results with the TN module ([PhysBench](https://github.com/KegangWangCCNU/PhysBench))
Please refer to the [source code](https://github.com/KegangWangCCNU/TemporalNormalization/blob/main/TN). 
When running the code, it needs to be integrated into [PhysBench](https://github.com/KegangWangCCNU/PhysBench).

![image](https://github.com/user-attachments/assets/61f92d53-448c-4ea1-8107-3c9e4f6f425c)

## Citation  
```
@misc{wang2024plugandplaytemporalnormalizationmodule,
      title={A Plug-and-Play Temporal Normalization Module for Robust Remote Photoplethysmography}, 
      author={Kegang Wang and Jiankai Tang and Yantao Wei and Mingxuan Liu and Xin Liu and Yuntao Wang},
      year={2024},
      eprint={2411.15283},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2411.15283}, 
}
```
