from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CelebA
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
from typing import Any, Callable, List, Optional, Tuple, Union

import os
import warnings
import torchvision.transforms as transforms


class CelebA_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=31):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 40))
        self.outlier_classes.remove(normal_class)

        train_set_full = MyCelebA(root=root, split='train', target_type='attr', download=True, transform=None, target_transform=None)

        MIN = []
        MAX = []
        for normal_classes in range(40):
            train_idx_normal = get_target_label_idx(train_set_full.train_labels.clone().data.cpu().numpy(), normal_classes)
            train_set = Subset(train_set_full, train_idx_normal)

            _min_ = []
            _max_ = []
            for idx in train_set.indices:
                gcm = global_contrast_normalization(train_set.dataset.data[idx].float(), 'l1')
                _min_.append(gcm.min())
                _max_.append(gcm.max())
            MIN.append(np.min(_min_))
            MAX.append(np.max(_max_))
        min_max = list(zip(MIN, MAX))

        # CelebA preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]] * 3,
                                                             [min_max[normal_class][1] - min_max[normal_class][0]] * 3)])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyCelebA(root=self.root, split='train', target_type='attr', download=True, transform=transform, target_transform=target_transform)
        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set.targets, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyCelebA(root=self.root, split='test', target_type='attr', download=True, transform=transform, target_transform=target_transform)


class MyCelebA(CelebA):
    """Torchvision CelebA class with patch of __getitem__ method to also return the index of a data sample."""

    # Allows for compatibility with deprecated code elsewhere in repository
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root: str, split: str = 'train', target_type: Union[List[str], str] = 'attr', transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False):
        super().__init__(root, split, target_type, transform, target_transform, download)

    def __getitem__(self, index):
        """Override the original method of the CelebA class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target, index  # only line changed
