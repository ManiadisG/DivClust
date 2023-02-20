# DivClust: Controlling Diversity in Deep Clustering

This is the official implementation of our CVPR 23' paper [DivClust: Controlling Diversity in Deep Clustering](https://arxiv.org/abs/2304.01042).

**Code coming soon.**

## Overview

![DivClust overview](/assets/overview.png "DivClust overview")
*Overview of DivClust*

DivClust is a method for controlling inter-clustering diversity in deep clustering frameworks. It consists of a novel loss that can be incorporated in most modern deep clustering frameworks in a straightforward way during their training, and which allows the user to specify their desired degree of inter-clustering diversity, which is then enforced  in the form of an upper bound threshold.

As shown in our [paper](https://arxiv.org/abs/2304.01042), DivClust adds minimal computational overhead and can significantly increase their performance with the use of off-the-shelf consensus clustering algorithms.

![DivClust overview](/assets/inter_clustering_similarity_viz.png "DivClust overview")
*Illustration of the inter-clustering similarity between 20 learned clusterings for various diversity targets. We report the diversity target D^T and measured diversity D^R, which are expressed in the NMI metric.*

## See also
[Original implementation of CC](https://github.com/Yunfan-Li/Contrastive-Clustering)

[Original implementation of PICA](https://github.com/Raymond-sci/PICA)

[Original implementation of IIC](https://github.com/xu-ji/IIC)

[Code for the consensus clustering algorithm SCCBG](http://doctor-nobody.github.io/codes/code_SCCBG.rar)

## Citation
If you use this repository, please cite:
```
@inproceedings{divclust2023cvpr,
    title={DivClust: Controlling Diversity in Deep Clustering},
    author={Metaxas, Ioannis Maniadis and Tzimiropoulos, Georgios and Patras, Ioannis},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2023},
}
```

## License

This project is licensed under the MIT License

## Acknowledgement

This work was supported by the EU H2020 AI4Media No. 951911 project.