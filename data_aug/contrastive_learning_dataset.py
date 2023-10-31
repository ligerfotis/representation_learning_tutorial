from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    @staticmethod
    def get_moco_pipeline_transform(size, s=1):
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        return data_transforms

    def get_dataset(self, name, n_views, type):
        if type == 'simclr' or type == 'simclr_q':
            transform = ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(32), n_views)
            transform_test = transforms.Compose([transforms.ToTensor()])
        elif type == 'moco':
            transform = ContrastiveLearningViewGenerator(self.get_moco_pipeline_transform(32), n_views)
            transform_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                      [0.2023, 0.1994, 0.2010])])
        valid_datasets = {'cifar10_train': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                                    transform=transform,
                                                                    download=True),
                          'cifar10_test': lambda: datasets.CIFAR10(self.root_folder, train=False,
                                                                   transform=transform_test,
                                                                   download=True),
                          'cifar10_memory': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                                     transform=transform_test,
                                                                     download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise NotImplementedError

        return dataset_fn()
