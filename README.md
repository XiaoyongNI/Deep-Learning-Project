# Deep-Learning-Project

## Instruction how to use the repo

### Dataset preparation 
Please refer to the readme file in Complex-yolo directory to prepare the dataset.
### Detector training
  Please using the train.sh file for detector training. Modify the --detector_type to be fine or coarse to train our fine and coarse detector separately.
 You can find the trained detectors: [coarse]https://1drv.ms/u/s!AveWDyBAPlGWg7BeG5IYBgy5Mi7XQg?e=jTsimx) and [fine](https://1drv.ms/u/s!AveWDyBAPlGWg7Bde_9LS4xl3ynmDw?e=yoaPe1).
![Training curve](./fig/d_train.png)


- Choose RegNetX-200MF parameters for RL
- Dataloader for RL
- RL training result(using old detectors):[here](https://drive.google.com/drive/folders/1KoteVPFJXtvmmRJVrNI6yzyM-apEZtdz?usp=sharing)
- RL training result using new detectors: [here](https://1drv.ms/u/s!AveWDyBAPlGWg7BM4mo_b1jgSGPiRw?e=1LjQJT)
### 
Get image id and patch id using the first column of coarse.txt and fine.txt
