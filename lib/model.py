from typing import Optional
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Accuracy
from torchvision import models
from pytorch_lightning.callbacks import ModelCheckpoint

from lib.loss import LogitNormLoss


class ImageNetModel(pl.LightningModule):
    def __init__(self, 
                backbone: str = 'resnet18',	
                lr: float = 0.001,
                num_classes: int = 3,
                optimizer: str = 'SGD',
                freeze_backbone: bool = False,
                logit_norm: bool = False,
                temperature: float = 1.0,
                save_dir:  str = 'checkpoints') -> None:
        """Image classification model. Loads different pretrained backbones from torchvision. 

        Args:
            backbone (str, optional): _description_. Defaults to 'resnet18'.
            lr (float, optional): _description_. Defaults to 0.001.
            num_classes (int, optional): _description_. Defaults to 3.
            optimizer (str, optional): _description_. Defaults to 'SGD'.
            freeze_backbone (bool, optional): _description_. Defaults to False.
            logit_norm (bool, optional): _description_. Defaults to False.
            temperature (float, optional): _description_. Defaults to 1.0.
            save_dir (str, optional): _description_. Defaults to 'checkpoints'.
        """
        super().__init__()

        # save hparams
        self.save_hyperparameters()

        # build model 
        self.__build_model()

        # define metric
        self.acc = Accuracy(task='multiclass', num_classes=self.hparams.num_classes)  


    def __build_model(self) -> None:
        """Define model layers & loss."""

        # load pretrained backbone
        model_func = getattr(models, self.hparams.backbone)
        backbone = model_func(pretrained=True)
        _in_features = backbone.fc.in_features
        
        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)

        # freeze backbone 
        if self.hparams.freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # add new classifier
        self.fc = nn.Linear(_in_features, self.hparams.num_classes)
        
        # define loss function
        if not self.hparams.logit_norm:
            self.loss_func = F.binary_cross_entropy_with_logits
        else:
            self.loss_func = LogitNormLoss(t=self.hparams.temperature)


    def forward(self, x):
        """Forward pass.

        Returns logits.
        """
        # feature extraction
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)

        # classification
        logits = self.fc(x)
        
        return logits


    def loss(self, logits, labels):
        """Compute the loss value."""
        return self.loss_func(input=logits, target=labels)
    
    def _shared_step(self, batch, batch_idx, stage):
        """Forward pass and compute loss."""
        
        # forward pass
        x, y = batch
        logits = self(x)
        
        # compute loss 
        y_true = F.one_hot(y, num_classes=self.hparams.num_classes).type(torch.float)
        loss = self.loss(logits, y_true)
        acc = self.acc(logits, y)
        
        # logging 
        self.log(f'{stage}/loss', loss, on_epoch=True)
        self.log(f'{stage}/acc', acc, on_epoch=True, prog_bar=True)
        
        return logits, loss, acc


    def training_step(self, batch, batch_idx):
        """Forward pass and compute loss."""
        logits, loss, acc = self._shared_step(batch, batch_idx, stage='train')
        return {'logits': logits, 'loss': loss, 'acc': acc}


    def validation_step(self, batch, batch_idx):         
        """Forward pass and compute loss."""
        logits, loss, acc = self._shared_step(batch, batch_idx, stage='val')
        return {'logits': logits, 'loss': loss, 'acc': acc}


    def test_step(self, batch, batch_idx):
        """Forward pass and compute loss."""
        logits, loss, acc = self._shared_step(batch, batch_idx, stage='test')
        return {'logits': logits, 'loss': loss, 'acc': acc}


    def predict_step(self, batch, batch_idx):
        """Predict step."""
        x, _ = batch
        logits = self.forward(x)
        y_conf, y_hat = torch.max(torch.softmax(logits, dim=1), dim=1)
        return y_conf, y_hat


    def configure_optimizers(self):
        """Configure optimizer."""
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        if self.hparams.optimizer == 'SGD':
            return torch.optim.SGD(trainable_parameters, lr=self.hparams.lr)
        elif self.hparams.optimizer == 'Adam':
            return torch.optim.Adam(trainable_parameters, lr=self.hparams.lr)
        elif self.hparams.optimizer == 'AdamW':
            return torch.optim.AdamW(trainable_parameters, lr=self.hparams.lr)
        else:
            raise NotImplementedError


