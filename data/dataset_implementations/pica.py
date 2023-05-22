from data.dataset_classes import ImageDataset
from data.transforms import PICATransforms
from data import dataset_readers


def _get_pica_datasets(data, labels, resize_shape, crop_size, norm_mean, norm_std, data_val=None, labels_val=None,
                       transforms_no=3):
    if data_val is None:
        data_val, labels_val = data, labels
    train_dataset = ImageDataset(data, labels, PICATransforms(resize_shape, crop_size, norm_mean, norm_std,
                                                              transforms_no=transforms_no))
    val_dataset = ImageDataset(data_val, labels_val,
                               PICATransforms(resize_shape, crop_size, norm_mean, norm_std, validation=True))
    return train_dataset, val_dataset


def cifar10_pica(dataset_path=None, transforms_no=3, *args, **kwargs):
    data, labels = dataset_readers.get_cifar10(dataset_path, ["merge"])
    train_dataset, val_dataset = _get_pica_datasets(data, labels, 40, 32, (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261), transforms_no=transforms_no)
    return train_dataset, val_dataset
