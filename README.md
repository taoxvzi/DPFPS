# DPFPS: Dynamic and Progressive Filter Pruning for Compressing Convolutional Neural Networks from Scratch
## The PyTorch implementation of DPFPS (AAAI 2021).
## Requirements
 Ubuntu Version: 18.04.2 LTS;

 Python Version: 3.6.9;

 Pytorch Version: 1.2.0;

 CUDA Version: 10.1.243.
## Run
### Test
The pruned models can be found in [Google Drive](https://drive.google.com/drive/folders/1OLTMgAvnEoDO9-_nsD2wqHLi-8LnKRv6).
```
python pruned_models_test.py --arch $architecture$  --model $pruned models$
```

### Train

```
bash ./scripts/run_resnet101_dpss_pr_0.45.sh
```

