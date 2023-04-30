import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from utils.misc import export_fn
import glob
import os
import numpy as np
from scipy import io as sio
from utils.misc import export_fn

### CIFAR10 ###

@export_fn
def get_cifar10(dataset_path, get_partition=("train", "val", "merge")):
    if dataset_path is None:
        dataset_path = './datasets/CIFAR10/'
    if "train" in get_partition or "merge" in get_partition:
        dtrain = CIFAR10(dataset_path, True, download=True)
        train_data = dtrain.data
        train_labels = np.array(dtrain.targets)
    if "val" in get_partition or "merge" in get_partition:
        dval = CIFAR10(dataset_path, False, download=True)
        val_data = dval.data
        val_labels = np.array(dval.targets)
    return_list = []
    if "train" in get_partition:
        return_list += [train_data, train_labels]
    if "val" in get_partition:
        return_list += [val_data, val_labels]
    if "merge" in get_partition:
        merged_data = np.concatenate((train_data.data, val_data.data))
        merged_labels = np.concatenate((train_labels, val_labels))
        return_list += [merged_data, merged_labels]
    return return_list

### CIFAR100 ###

@export_fn
def get_cifar100(dataset_path, get_partition=("train", "val", "merge"), superclasses=False):
    if dataset_path is None:
        dataset_path = './datasets/CIFAR100/'
    if "train" in get_partition or "merge" in get_partition:
        dtrain = CIFAR100(dataset_path, train=True, download=True)
        if superclasses:
            dtrain = _cifar100_class_to_superclass(dtrain)
        train_data = dtrain.data
        train_labels = np.array(dtrain.targets)
    if "val" in get_partition or "merge" in get_partition:
        dval = CIFAR100(dataset_path, train=False, download=True)
        if superclasses:
            dval = _cifar100_class_to_superclass(dval)
        val_data = dval.data
        val_labels = np.array(dval.targets)
    return_list = []
    if "train" in get_partition:
        return_list += [train_data, train_labels]
    if "val" in get_partition:
        return_list += [val_data, val_labels]
    if "merge" in get_partition:
        merged_data = np.concatenate((train_data.data, val_data.data))
        merged_labels = np.concatenate((train_labels, val_labels))
        return_list += [merged_data, merged_labels]
    return return_list


def _cifar100_class_to_superclass(dataset):
    aquatic_mammals = ['beaver', 'dolphin', 'otter', 'seal', 'whale']
    fish = ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']
    flowers = ['orchid', 'poppy', 'rose', 'sunflower', 'tulip']
    food_containers = ['bottle', 'bowl', 'can', 'cup', 'plate']
    fruits_and_vegetables = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']
    house_electrical_devices = ['clock', 'keyboard', 'lamp', 'telephone', 'television']
    household_furniture = ['couch', 'bed', 'chair', 'table', 'wardrobe']
    insects = ['bee', 'butterfly', 'beetle', 'caterpillar', 'cockroach']
    large_carnivores = ['bear', 'leopard', 'lion', 'tiger', 'wolf']
    large_man_made_outdoor_things = ['bridge', 'castle', 'house', 'road', 'skyscraper']
    large_natural_outdoor_scenes = ['cloud', 'forest', 'mountain', 'plain', 'sea']
    large_omnivores_herbivores = ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']
    medium_sized_mammals = ['fox', 'porcupine', 'possum', 'raccoon', 'skunk']
    non_insect_invertebrates = ['crab', 'lobster', 'snail', 'spider', 'worm']
    people = ['baby', 'boy', 'girl', 'man', 'woman']
    reptiles = ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle']
    small_mammals = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
    trees = ['maple_tree', 'oak_tree', 'pine_tree', 'palm_tree', 'willow_tree']
    vehicles_1 = ['bicycle', 'bus', 'pickup_truck', 'motorcycle', 'train']
    vehicles_2 = ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']

    superclasses_lists = [aquatic_mammals, fish, flowers, food_containers, fruits_and_vegetables,
                          house_electrical_devices, household_furniture, insects, large_carnivores,
                          large_man_made_outdoor_things, large_natural_outdoor_scenes, large_omnivores_herbivores,
                          medium_sized_mammals, non_insect_invertebrates, people, reptiles, small_mammals, trees,
                          vehicles_1, vehicles_2]
    superclasses_names = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                          'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                          'large man-made outdoor things', 'large natural outdoor scenes',
                          'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates',
                          'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']

    class_2_superclass = {}
    for subclasses, superclass in zip(superclasses_lists, superclasses_names):
        for v in subclasses:
            class_2_superclass[v] = superclass

    class_to_idx = {}
    i = 0
    for k in class_2_superclass.values():
        if k not in class_to_idx.keys():
            class_to_idx[k] = i
            i += 1
    dataset_dict = dataset.class_to_idx
    idx_to_supclass_idx = {}
    for k, v in dataset_dict.items():
        idx_to_supclass_idx[v] = class_to_idx[class_2_superclass[k]]
    for i in range(len(dataset.targets)):
        dataset.targets[i] = idx_to_supclass_idx[dataset.targets[i]]
    return dataset


### ImageNet ###

@export_fn
def get_imagenet(dataset_path=None, version="default"):
    if dataset_path is None:
        dataset_path = "./datasets/ImageNet"
    if os.path.isdir(dataset_path + '/ILSVRC2012_devkit_t12') and os.path.isdir(dataset_path + '/val') and os.path.isdir(dataset_path + '/train'):
        return get_imagenet_file_reading(dataset_path, version)
    elif os.path.isfile(dataset_path + '/train.h5') and os.path.isfile(dataset_path + '/val.h5'):
        return dataset_path + '/train.h5', None, dataset_path + '/val.h5', None
    

def get_imagenet_file_reading(dataset_path, version):
    idx_to_wnid, wnid_to_idx, wnid_to_classes = _parse_meta_mat(dataset_path + '/ILSVRC2012_devkit_t12')
    if version !="default":
        if version == "imagenet-dogs":
            c_folder = ["n02085936", "n02086646", "n02088238", "n02091467", "n02097130", "n02099601", "n02101388",
                        "n02101556", "n02102177", "n02105056", "n02105412", "n02105855", "n02107142", "n02110958",
                        "n02112137"]
        elif version=="imagenet-10":
            c_folder = ["n02056570", "n02085936", "n02128757", "n02690373", "n02692877", "n03095699", "n04254680",
                        "n04285008", "n04467665", "n07747607"]
        idx_to_wnid_, wnid_to_idx_, wnid_to_classes_ = {},{},{}
        for i,c_ in enumerate(c_folder):
            idx_to_wnid_[i]=c_
            wnid_to_idx_[c_]=i
            wnid_to_classes_[c_]=wnid_to_classes[c_]
        idx_to_wnid, wnid_to_idx, wnid_to_classes = idx_to_wnid_, wnid_to_idx_, wnid_to_classes_
    train_path = dataset_path + '/train/'
    train_samples, train_labels = [], []
    for k in wnid_to_classes.keys():
        k_samples = glob.glob(train_path + k + '/*')
        train_samples += k_samples
        train_labels += len(k_samples) * [wnid_to_idx[k]]

    val_labels = _parse_val_groundtruth_txt(dataset_path + '/ILSVRC2012_devkit_t12')
    val_path = dataset_path + '/val'
    val_samples = glob.glob(val_path + '/*')
    val_samples.sort()

    return np.array(train_samples), np.array(train_labels), np.array(val_samples), np.array(val_labels)

def _parse_val_groundtruth_txt(devkit_root):
    file = os.path.join(devkit_root, "data",
                        "ILSVRC2012_validation_ground_truth.txt")
    with open(file, 'r') as txtfh:
        val_idcs = txtfh.readlines()
    return [int(val_idx) - 1 for val_idx in val_idcs]


def _parse_meta_mat(devkit_root):
    metafile = os.path.join(devkit_root, "data", "meta.mat")
    meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
            if num_children == 0]
    idcs, wnids, classes = list(zip(*meta))[:3]
    classes = [tuple(clss.split(', ')) for clss in classes]
    idx_to_wnid = {idx - 1: wnid for idx, wnid in zip(idcs, wnids)}
    wnid_to_idx = {wnid: idx - 1 for idx, wnid in zip(idcs, wnids)}
    wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}

    return idx_to_wnid, wnid_to_idx, wnid_to_classes