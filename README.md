# Deep-Learning-Project

## Instruction how to use the repo

### Dataset preparation 
Please refer to the readme file in Complex-yolo directory to prepare the dataset.
### Detector training

- RL modification from python2 to python3
- prepare yolo and the heatmap(potentially useful for RL input)
- You can find the trained detectors: [coarse](https://1drv.ms/u/s!AveWDyBAPlGWg7BAA1oJnvywOB_WBw?e=bDqar5) and [fine](https://1drv.ms/u/s!AveWDyBAPlGWg7BBYau3uR2qHUAlOQ?e=hIkykF).

- Choose RegNetX-200MF parameters for RL
- Dataloader for RL
- RL training result(using old detectors):[here](https://drive.google.com/drive/folders/1KoteVPFJXtvmmRJVrNI6yzyM-apEZtdz?usp=sharing)
- RL training result using new detectors: [here](https://1drv.ms/u/s!AveWDyBAPlGWg7BM4mo_b1jgSGPiRw?e=1LjQJT)
### 
Get image id and patch id using the first column of coarse.txt and fine.txt
