from torch.utils.data import Subset, ConcatDataset
from PIL import Image
from torchvision.datasets import CelebA
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
from typing import Any, Callable, List, Optional, Tuple, Union

import os
import warnings
import math
import torchvision.transforms as transforms


class CelebA_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=31):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 54))
        if normal_class != -1:
            self.outlier_classes.remove(normal_class)

        # Precomputed GCN over all images in CelebA
        min_max = (-9.556149, 16.703974)

        # CelebA preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        # No contrast normalization enabled by default: contrast is too poor to learn CelebA features
                                        #transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        #transforms.Normalize([min_max[0]] * 3,
                                        #                     [min_max[1] - min_max[0]] * 3),
#                                        transforms.CenterCrop(178),
                                        transforms.Resize(size=(160, 160))])

        target_transform = transforms.Lambda(lambda x: int(x[normal_class] == 0))

        if normal_class == -1:
            self.train_set = MyCelebA(root=self.root, split='train', target_type='attr', download=False, transform=transform, target_transform=target_transform)
            valid_set = MyCelebA(root=self.root, split='valid', target_type='attr', download=False, transform=transform, target_transform=target_transform)
            test_only_set = MyCelebA(root=self.root, split='test', target_type='attr', download=False, transform=transform, target_transform=target_transform)
            self.test_set = ConcatDataset([test_only_set, valid_set])
        else:
            all_set = MyCelebA(root=self.root, split='all', target_type='attr', download=False, transform=transform, target_transform=target_transform) 
            idx_normal = get_target_label_idx(all_set.attr[:,normal_class], [1])
            idx_normal_train = idx_normal[:math.floor(0.8 * len(idx_normal))]
            idx_normal_test = idx_normal[math.floor(0.8 * len(idx_normal)):]
            self.train_set = Subset(all_set, idx_normal_train)
            self.test_set = Subset(all_set, idx_normal_test)

class MyCelebA(CelebA):
    """Torchvision CelebA class with patch of __getitem__ method to also return the index of a data sample."""
    apply_target_transform = True

    file_list = [
        # File ID                                      MD5 Hash                            Filename
        ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "90b6bf6da05dd8b6c60e2731a854b908", "list_attr_celeba.txt"),
        ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

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

            if self.apply_target_transform:
                if self.target_transform is not None:
                    target = self.target_transform(target)
        else:
            target = None

        return X, target, index  # only line changed
