# Label-Propagation-with-Augmented-Anchors (A2LP)

Official codes of the ECCV2020 spotlight (label propagation with augmented anchors: a simple semi-supervised learning
 baseline for unsupervised domain adaptation) [[Paper](https://arxiv.org/abs/2007.07695)].
In this work, we investigating SSL principles for UDA problems.

## One-sentence Summary
Proper algorithmic adaptation should be made when applying the SSL techniques to UDA tasks, even both tasks of UDA and SSL adopt the labeled and unlabeled data as the input.


## Usage
    Please refer to the 'run.sh'. We also provide the corresponding log file in the file of './test'
    You can start with these examples easily.  
    
    NOTE: Results are based on the ImageNet pre-trained features !!! No additional training involved. 
    
    Note that the A2LP can introduce excellent pseudo labels of unlabeled target data in DA 
    (compared to the FC-based classifier and the clustering algorithm). Therefore it could
    empower algorithms of DA using pseudo labels of unlabeled target data.

## Requirement
1. PyTorch 1.2.0
2. [spherecluster](https://github.com/jasonlaska/spherecluster)
3. [nndescent](https://github.com/lmcinnes/pynndescent)

## Dataset
The structure of the dataset should be like

```
Office-31
|_ amazon
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ...
|_ dslr
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ...
|_ ...
```

## Citation
    @inproceedings{zhang2020label,
      title={Label Propagation with Augmented Anchors: A Simple Semi-Supervised Learning baseline for Unsupervised Domain Adaptation},
      author={Zhang, Yabin and Deng, Bin and Jia, Kui and Zhang, Lei},
      booktitle={European Conference on Computer Vision},
      pages={781--797},
      year={2020},
      organization={Springer}
    }


## Contact
    If you have any problem about our code, feel free to contact
    - zhang.yabin@mail.scut.edu.cn
    
    or describe your problem in Issues. 
