# Inferring the Class Conditional Response Map for Weakly Supervised Semantic Segmentation

This repository contains the code for inferring better class activation maps from a classifier without re-training.
With a trained classification networks, this method pushs the class activation maps to cover more object areas without any network training, which may facilitate down-stream weakly supervised semantic segmentation and object localization.
For example:

<img width="591" alt="image" src="https://user-images.githubusercontent.com/13931546/141668663-0979490e-9ec9-45e2-bef6-56f6cbdc408d.png">


##
If you use this code in an academic context, please cite the following references:

        @inproceedings{sun2022inferring,
          title={Inferring the Class Conditional Response Map for Weakly Supervised Semantic Segmentation},
          author={Sun, Weixuan and Zhang, Jing and Barnes, Nick},
          booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
          pages={2878--2887},
          year={2022}
        }

## Enviroment:


## Instructions:
First, run the baseline cam inference to obtain the mass center of every activation regions, then split the image into patches according to the mass center:
**classification weight for Pascal voc can be obtained from:[psa](https://github.com/jiwoon-ahn/psa) or  https://1drv.ms/u/s!Ak3sXyXVg7818CLKis4D2CXKXV6D?e=0k9HWo**


    python split_img.py --weights [Your classification weights path] --voc12_root [Pascal VOC root path]   --split_path [path to save the splitted image] --heatmap [If you want to visualize the baseline CAM] 

  We provide the splitted images for PASCAL VOC dataset(so step one could be skipped):
 https://1drv.ms/u/s!Ak3sXyXVg7818CSGY3V0Th4hZiak?e=bpaqkw


Second, run the inference code to generate refined class activation maps: 
    
    python infer_cam.py --weights [Your classification weights path] --split_path [The path of the splitted images] --out_cam [Path to save the output CAM] --heatmap [If you want to visualize the refined CAM] 

You can replace this network with any other pre-trained networks and obtain corresponding class activation maps without re-training the network.
    
## Pseudo label and semantic segmentation training
Refinement: We adopt the random walk method via affinity to refine the map as pixel-wise pseudo ground truths for semantic segmentation. Please refer to [psa](https://github.com/jiwoon-ahn/psa)


##

> Thanks for the code provided by [psa](https://github.com/jiwoon-ahn/psa)

> Note: This work is accepted to WACV 2022, and was originally proposed in November/2020.


## Reference:
1. Jiwoon Ahn and Suha Kwak. Learning pixel-level semantic affinity with image-level supervision for weakly supervised semantic segmentation. CVPR, 2018.



