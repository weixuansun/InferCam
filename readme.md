# Inferring the Class Conditional Response Map for Weakly Supervised Semantic Segmentation

This repository contains the code for inferring better class activation maps from a classifier without re-training.
With a trained classification networks, this method pushs the class activation maps to cover more object areas, which may facilitate down-stream weakly supervised semantic segmentation and object localization.
For example:


##
If you use this code in an academic context, please cite the following references:

Inferring the Class Conditional Response Map for Weakly Supervised Semantic Segmentation
W Sun, J Zhang, N Barnes - arXiv preprint arXiv:2110.14309, 2021

## Enviroment:


## Instructions:
First, run the baseline cam inference to obtain the mass center of every activation regions, then split the image into patches according to the mass center:
**classification weight for Pascal voc can be obtained from:  https://1drv.ms/u/s!Ak3sXyXVg7818CLKis4D2CXKXV6D?e=0k9HWo**


    python split_img.py --weights [Your classification weights path] --voc12_root [Pascal VOC root path]   --split_path [path to save the splitted image] --heatmap [If you want to visualize the baseline CAM] 

  We provide the splitted images for PASCAL VOC dataset:
  splitted image link: https://1drv.ms/u/s!Ak3sXyXVg7818CSGY3V0Th4hZiak?e=bpaqkw


Second, run the inference code to generate refined class activation maps: 
    
    python infer_cam.py --weights [Your classification weights path] --split_path [The path of the splitted images] --out_cam [Path to save the output CAM] --heatmap [If you want to visualize the refined CAM] 
    
## Pseudo label and semantic segmentation training
Refinement: We adopt the random walk method via affinity to refine the map as pixel-wise pseudo ground truths for semantic segmentation. Please refer to [psa](https://github.com/jiwoon-ahn/psa)


##

> Thanks for the code provided by [psa](https://github.com/jiwoon-ahn/psa)

> Note: This work is accepted to WACV 2022, and was originally proposed in November/2020.





