# Introduction
This repository is for ***Audio-Guided Attention Network for Weakly Supervised Violence Detection*** (ICCECE 2022). The original paper can be found [here](https://ieeexplore.ieee.org/document/9712793). Please contact me by email if you have any further queries.
 
## Training Stage
- Download the extracted I3D features of XD-Violence dataset from [here](https://roc-ng.github.io/XD-Violence/).
- Change the file paths of ```make_list.py``` in the list folder to generate the training and test list.
- Change the hyperparameters in ```option.py```, where we keep default settings as mentioned in our paper.
- Run the following command for training:
```
python main.py
```
## Test Stage
- Change the checkpoint path of ```infer.py```.
- Run the following command for test:
```
python infer.py
```

## Acknowledgements
The implementation mainly references the repo of [XDVioDet](https://github.com/Roc-Ng/XDVioDet). We greatly appreciate their excellent contribution.
