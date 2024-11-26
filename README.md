# A Plug-and-Play Temporal Normalization Module for Robust Remote Photoplethysmography
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

## Inserting TN into any end-to-end model (Keras)
Please refer to the source code [TN/model_tn.py](https://github.com/KegangWangCCNU/TemporalNormalization/blob/main/TN/model_tn.py).

## Implementation ([rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)) 
Please refer to [PhysNet_TN_rPPG-Toolbox.py](https://github.com/KegangWangCCNU/TemporalNormalization/blob/main/PhysNet_TN_rPPG-Toolbox.py) and [Example_rPPG-Toolbox.yaml](https://github.com/KegangWangCCNU/TemporalNormalization/blob/main/Example_rPPG-Toolbox.yaml) in the source code.
`Example_rPPG-Toolbox.yaml` is its configuration file.  

* __Note: For rPPG-Toolbox, do not use any preprocessing or postprocessing, and ensure that both the input and output in the yaml are set to 'Raw'.__

## Benchmark results with the TN module ([PhysBench](https://github.com/KegangWangCCNU/PhysBench))
Please refer to the source code [TN folder](https://github.com/KegangWangCCNU/TemporalNormalization/blob/main/TN). 

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
