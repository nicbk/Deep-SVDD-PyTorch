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

        min_max = [(-5.549275, 9.29441), (-9.556149, 8.864991), (-5.953659, 7.4443283), (-6.4589705, 8.864991), (-6.01108, 8.693999), (-9.556149, 9.282144), (-6.251924, 8.97718), (-6.4589705, 9.498133), (-4.918437, 9.498133), (-8.171633, 6.0304255), (-5.811125, 11.721458), (-4.5102797, 9.158465), (-6.2917924, 8.071238), (-6.2917924, 10.04199), (-6.6773515, 8.340185), (-6.2917924, 9.498133), (-6.806502, 12.566853), (-7.5726933, 7.733251), (-8.171633, 6.6883006), (-5.785223, 6.6513276), (-6.2917924, 9.498133), (-6.2917924, 7.0371265), (-5.538945, 12.299471), (-6.7143073, 8.987023), (-7.5726933, 9.812891), (-8.171633, 6.9903736), (-9.556149, 6.6022143), (-7.5726933, 7.744242), (-7.0190334, 11.603071), (-7.0190334, 7.1444798), (-6.1495414, 9.764772), (-5.785223, 9.158465), (-9.556149, 6.770698), (-5.785223, 9.282144), (-7.7556286, 7.432579), (-5.3981357, 11.361711), (-7.5726933, 7.7363286), (-7.5726933, 7.8059897), (-6.2917924, 12.299471), (-7.5726933, 9.812891)]

        # CelebA preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]] * 3,
                                                             [min_max[normal_class][1] - min_max[normal_class][0]] * 3)])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyCelebA(root=self.root, split='train', target_type='attr', download=False, transform=transform, target_transform=target_transform)
        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set.attr[:,self.normal_classes], 1)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyCelebA(root=self.root, split='test', target_type='attr', download=False, transform=transform, target_transform=target_transform)


class MyCelebA(CelebA):
    """Torchvision CelebA class with patch of __getitem__ method to also return the index of a data sample."""

    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

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
