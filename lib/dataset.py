import os 
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from typing import Callable, Dict, List, Optional, Tuple, Union



class SubImageNet(Dataset):
    def __init__(self, 
                img_dir: str, 
                split: str,
                wnid_to_class: Dict[str, int],
                class_to_label: Dict[int, str],
                transforms: Optional[Callable] = None
                ) -> None:

        self.img_dir = img_dir
        self.split = split  
        self.wnid_to_class = wnid_to_class
        self.class_to_label = class_to_label
        self.transforms = transforms

        samples = self.make_dataset(self.img_dir, self.split, self.wnid_to_class)

        self.classes = list(self.class_to_label.values())
        self.samples = samples
        self.targets = [s[1] for s in samples]



    def make_dataset(self, directory: str, split: str, class_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path/to/sample, class).

        Args:
            directory (str): root dataset directory.
            class_to_idx (dict[str, int]): dictionary mapping class name to class index.

        Returns:
            list[tuple[str, int]]: samples of a form (path_to_sample, class).
        """
        samples = []
        for class_name in sorted(class_to_idx.keys()):
            class_idx = class_to_idx[class_name]
            class_dir = os.path.join(directory, split, class_name)
            if not os.path.isdir(class_dir):
                continue
            for root, _, files in sorted(os.walk(class_dir)):
                for file in sorted(files):
                    path = os.path.join(root, file)
                    item = path, class_idx
                    samples.append(item)
        return samples
    
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, target = self.samples[index]
        image = self.load_image(path)
        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed['image']
            image = torch.from_numpy(image).permute(2, 0, 1).type(torch.float32) 
        else:
            image = torch.from_numpy(image / 255.).permute(2, 0, 1).type(torch.float32) 
        target = torch.tensor(target)
        return image, target


    def load_image(self, path: str) -> Image.Image:
        """Loads an image in RGB format using pillow.

        Args:
            path (str): path to file. 

        Returns:
            Image.Image: Image in RGB format. 
        """
        with open(path, "rb") as f:
            img = Image.open(f)
            return np.asarray(img.convert("RGB"))


    def __len__(self) -> int:
        return len(self.samples)


    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        images = []
        targets = []
        for b in batch:
            images.append(b[0])
            targets.append(b[1])
        images = torch.stack(images, dim=0)
        targets = torch.stack(targets, dim=0)
        return images, targets
    




class CatStudyDataset(SubImageNet):

    PATH_TO_LABEL = {
        1: 'Hauskatze',
        2: 'Wildkatze',
        3: 'Großkatze',
        4: 'Hauskatze',
        5: 'Wildkatze',
        6: 'Großkatze'
    }

    LABEL_TO_IDX = {
        'Hauskatze': 0,
        'Wildkatze': 1,
        'Großkatze': 2
    }

    IDX_TO_LABEL = {
        0: 'Hauskatze',
        1: 'Wildkatze',
        2: 'Großkatze'
    }

    def __init__(
            self, 
            img_dir: str, 
            split: str,
            transforms: Optional[Callable] = None
    ) -> None:
        self.img_dir = img_dir
        self.split = split
        self.transforms = transforms

        samples = self.make_dataset()

        self.classes = list(self.LABEL_TO_IDX.values())
        self.samples = samples
        self.targets = [s[1] for s in samples]
    


    def make_dataset(self) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path/to/sample, class).

        Args:
            directory (str): root dataset directory.

        Returns:
            list[tuple[str, int]]: samples of a form (path_to_sample, class).
        """

        directory = os.path.join(self.img_dir, self.split)

        samples = []
        for root, _, files in sorted(os.walk(directory)):
            for file in sorted(files):
                path = os.path.join(root, file)
                label = self.read_label(path)
                class_idx = self.LABEL_TO_IDX[label]
                item = path, class_idx
                samples.append(item)
        return samples


    def read_label(self, file: str) -> int:
        """Returns the cat label from the filename.

        Args:
            file (str): filename.

        Raises:
            ValueError: If lab3el cannot be recognized from the filename. 

        Returns:
            int: Class label for the cat image.
        """
        idx = int(file[-7])
        if idx not in list(self.PATH_TO_LABEL.keys()):
            raise ValueError('Class not recognized for idx {}'.format(idx))
        label = self.PATH_TO_LABEL[idx]
        return label



        






