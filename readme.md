# Inferring the Class Conditional Response Map for Weakly Supervised Semantic Segmentation

This repository contains the code for inferring better class activation maps from a classifier without re-training.
With a trained classification networks, this method pushs the class activation maps to cover more object areas, which may facilitate down-stream weakly supervised semantic segmentation and object localization.
For example:


## Reference:
If you use this code in an academic context, please cite the following references:

Inferring the Class Conditional Response Map for Weakly Supervised Semantic Segmentation
W Sun, J Zhang, N Barnes - arXiv preprint arXiv:2110.14309, 2021


## Instructions:
First, run the baseline cam inference to obtain the mass center of every activation regions, then split the image into patches according to the mass center:
classification weight for Pascal voc can be obtained from:  https://1drv.ms/u/s!Ak3sXyXVg7818CLKis4D2CXKXV6D?e=0k9HWo


splitted image link: https://1drv.ms/u/s!Ak3sXyXVg7818CSGY3V0Th4hZiak?e=bpaqkw

Second, run the inference code to generate refined class activation maps:




