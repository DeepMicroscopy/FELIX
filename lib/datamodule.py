import pytorch_lightning as pl
import torch 
import torchvision.transforms as T
import albumentations as A


from typing import Dict, List, Optional
from torch.utils.data import DataLoader

from .dataset import SubImageNet, CatStudyDataset


class ImageNetModule(pl.LightningDataModule):
    def __init__(self,
                img_dir: str, 
                wnid_to_class: Dict[str, int],
                class_to_label: Dict[int, str],
                patch_size: int = 224,
                batch_size: int = 32,
                num_workers: int = 4,
                pilot_data: bool = False
            ) -> None:
        """ImageNet data module for training an image classfication model. 

        Args:
            img_dir (str): Path to image directory with subdirectories for train/val/test folder.
            wnid_to_class (Dict[str, int]): Dictionary mapping WNID to class index of subset of ImageNet classes.
            class_to_label (Dict[int, str]): Dictionary mapping class index to label of subset of ImageNet classes.
            patch_size (int, optional): Patch size. Defaults to 224.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of processes. Defaults to 4.
            pilot_data (bool, optional): Whether to evaluate on the CatStudyData.
        """
        super().__init__()
        self.img_dir = img_dir
        self.wnid_to_class = wnid_to_class
        self.class_to_label = class_to_label
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pilot_data = pilot_data


    def setup(self, stage: str = None) -> None:
        """Setup train/val/test splits"""

        if stage == 'fit':
            self.train_dataset = SubImageNet(
                img_dir=self.img_dir, 
                split='train',
                wnid_to_class=self.wnid_to_class,
                class_to_label=self.class_to_label,
                transforms=self.train_transform
                )
            
            if not self.pilot_data:
                self.val_dataset = SubImageNet(
                    img_dir=self.img_dir, 
                    split='val',
                    wnid_to_class=self.wnid_to_class,
                    class_to_label=self.class_to_label,
                    transforms=self.valid_transform
                    )
            else:
                self.val_dataset = CatStudyDataset(
                    train=True,
                    transforms=self.valid_transform
                    )

    
        
        if stage == 'test':

            if not self.pilot_data:
                self.test_dataset = SubImageNet(
                    img_dir=self.img_dir, 
                    split='test',
                    wnid_to_class=self.wnid_to_class,
                    class_to_label=self.class_to_label,
                    transforms=self.valid_transform
                    )
            else:
                self.test_dataset = CatStudyDataset(
                    train=False,
                    transforms=self.valid_transform
                    )



        if stage == 'predict' or stage is None:
            if not self.pilot_data:
                self.predict_dataset = SubImageNet(
                    img_dir=self.img_dir, 
                    split='test',
                    wnid_to_class=self.wnid_to_class,
                    class_to_label=self.class_to_label,
                    transforms=self.valid_transform
                    )
            else:
                self.predict_dataset = CatStudyDataset(
                    train=False,
                    transforms=self.valid_transform
                    )


    @property
    def normalize_transform(self):
        return A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return A.Compose([   
                    A.Resize(self.patch_size, self.patch_size),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    A.RandomResizedCrop(height=self.patch_size, width=self.patch_size, p=0.3),
                    A.RandomRotate90(p=0.3),
                    A.GaussianBlur(blur_limit=(3,7), sigma_limit=0, p=0.3),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    self.normalize_transform,
                    A.Cutout(num_holes=16, max_h_size=64, max_w_size=64, p=0.1)
                ])

    @property
    def valid_transform(self):
        return A.Compose([
            A.Resize(self.patch_size, self.patch_size),
            self.normalize_transform
            ])


    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                        batch_size=self.batch_size, 
                        shuffle=True, 
                        num_workers=self.num_workers, 
                        collate_fn=self.train_dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                        batch_size=self.batch_size, 
                        shuffle=False, 
                        num_workers=self.num_workers, 
                        collate_fn=self.val_dataset.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                        batch_size=self.batch_size, 
                        shuffle=False, 
                        num_workers=self.num_workers, 
                        collate_fn=self.test_dataset.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, 
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=self.num_workers,
                        collate_fn=self.predict_dataset.collate_fn)






class CatStudyModule(pl.LightningDataModule):
    def __init__(
            self,
            img_dir: str, 
            patch_size: int = 224,
            batch_size: int = 32,
            num_workers: int = 4,
            ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.excluded_dataset = self.create_dataset(split='Excluded')
        self.validation_dataset = self.create_dataset(split='Validation')
        self.experimental_dataset = self.create_dataset(split='Experimental')

    
    def set_split(self, split: str) -> None:
        """_summary_

        Args:
            split (str): _description_
        """
        self.split = split 

    

    def create_dataset(self, split: str) -> CatStudyDataset:
        """_summary_

        Args:
            split (str): _description_

        Returns:
            CatStudyDataset: _description_
        """
        return CatStudyDataset(
            img_dir=self.hparams.img_dir,
            split=split,
            transforms=self.valid_transform
        )
    

    def predict_dataloader(self) -> DataLoader:
        if self.split == 'Excluded':
            return DataLoader(self.excluded_dataset, self.hparams.batch_size, num_workers=self.hparams.num_workers)
        elif self.split == 'Experimental':
            return DataLoader(self.experimental_dataset, self.hparams.batch_size, num_workers=self.hparams.num_workers)
        else:
            return DataLoader(self.validation_dataset, self.hparams.batch_size, num_workers=self.hparams.num_workers)


    @property
    def normalize_transform(self):
        return A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    @property
    def valid_transform(self):
        return A.Compose([
            A.Resize(self.hparams.patch_size, self.hparams.patch_size),
            self.normalize_transform
            ])
