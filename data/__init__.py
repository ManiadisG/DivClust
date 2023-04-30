

def build_dataset(clustering_framework: str, data: str, dataset_path:str, args):
    
    if clustering_framework.lower()=="cc":
        from data.dataset_implementations import cc as cc
        if data.lower()=="cifar10":
            return cc.cifar10_cc(dataset_path, args.crop_size)
        elif data.lower()=="cifar100":
            return cc.cifar100_cc(dataset_path, args.crop_size)
        elif data.lower()=="imagenet_dogs":
            return cc.imagenet_dogs_cc(dataset_path, args.crop_size)
        elif data.lower()=="imagenet_10":
            return cc.imagenet_10_cc(dataset_path, args.crop_size)
    elif clustering_framework.lower()=="iic" or clustering_framework.lower()=="pica":
        from data.dataset_implementations import pica
        if data.lower()=="cifar10":
            return pica.cifar10_pica(dataset_path)
    raise ValueError("Unknown dataset")


        