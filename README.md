# DivClust: Controlling Diversity in Deep Clustering

This is the official implementation of our CVPR 23' paper [DivClust: Controlling Diversity in Deep Clustering](https://arxiv.org/abs/2304.01042).

## Overview

![DivClust overview](/assets/overview.png "DivClust overview")
*Overview of DivClust*

DivClust is a method for controlling inter-clustering diversity in deep clustering frameworks. It consists of a novel loss that can be incorporated in most modern deep clustering frameworks in a straightforward way during their training, and which allows the user to specify their desired degree of inter-clustering diversity, which is then enforced  in the form of an upper bound threshold.

As shown in our [paper](https://arxiv.org/abs/2304.01042), DivClust adds minimal computational overhead and can significantly increase their performance with the use of off-the-shelf consensus clustering algorithms.

![DivClust overview](/assets/inter_clustering_similarity_viz.png "DivClust overview")
*Illustration of the inter-clustering similarity between 20 learned clusterings for various diversity targets. We report the diversity target D^T and measured diversity D^R, which are expressed in the NMI metric.*

## Usage

To create an environment, execute the following commands:

```
conda create -n divclust_env python=3.8 &&
conda activate divclust_env &&
conda config --add channels conda-forge &&
conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch &&
pip install scipy wandb PyYAML scikit-learn termcolor matplotlib opencv-contrib-python
```

To run experiments, this repository uses the .yaml files in the ```configs``` directory. For each run, arguments are read from ```configs/main_config.yaml```, and are then supplemented (and in case of conflicts overwritten) by the secondary config files in that dir, which is identified with the ```preset``` argument. Arguments can further be provided in bash.

For example, to run an experiment with the CC deep clustering framework with 2 clusterings with a diversity target of $D^T=0.8$, one would run the following command:

```
python main.py --preset cc_cifar10 --clusterings 2 --NMI_target 0.8
```

## See also
[Original implementation of CC](https://github.com/Yunfan-Li/Contrastive-Clustering)

[Original implementation of PICA](https://github.com/Raymond-sci/PICA)

[Original implementation of IIC](https://github.com/xu-ji/IIC)

[Code for the consensus clustering algorithm SCCBG](http://doctor-nobody.github.io/codes/code_SCCBG.rar)

We thank the authors of the above for making their code public.

## Citation
If you use this repository, please cite:
```
@inproceedings{divclust2023,
  title={DivClust: Controlling Diversity in Deep Clustering},
  author={Metaxas, Ioannis Maniadis and Tzimiropoulos, Georgios and Patras, Ioannis},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3418--3428},
  year={2023}
}
```

## License

This project is licensed under the MIT License

## Acknowledgement

This work was supported by the EU H2020 AI4Media No. 951911 project.