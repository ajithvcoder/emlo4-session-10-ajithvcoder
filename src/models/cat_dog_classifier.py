import lightning as L
import timm
import torch
import torch.nn.functional as F
from torch import optim
from torchmetrics import Accuracy, MaxMetric
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import os
import ast
from glob import glob

class CatDogClassifier(L.LightningModule):
    def __init__(self, base_model: str = "resnet18", pretrained=False, num_classes: int = 10,
        lr: float = 1e-3, weight_decay: float = 0.0001, **kwargs):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.patch_size = kwargs['patch_size']
        self.embed_dim = kwargs['embed_dim']
        print(kwargs)
        # Load pre-trained ResNet18 model
        self.model = timm.create_model(base_model, pretrained=pretrained, 
            num_classes=self.num_classes, dims=ast.literal_eval(kwargs['dims']), depths=ast.literal_eval(kwargs['depths'])
            )

        # Multi-class accuracy with num_classes=2
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc_best = MaxMetric()
        self.validation_step_outputs = []
        self.train_confusion =  MulticlassConfusionMatrix(num_classes=2)
        self.test_confusion =  MulticlassConfusionMatrix(num_classes=2)

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.train_acc(preds, y)
        if self.current_epoch == self.trainer.max_epochs - 1:
            self.train_confusion(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def on_train_end(self):        
        print('CONFUSION_MATRIX:', self.train_confusion.compute())
        fig_, ax_ = self.train_confusion.plot()
        
        # Find the most recent metrics.csv file
        eval_log_files = glob("logs/train/multiruns/*/*/train.log")
        latest_eval_log = max(eval_log_files, key=os.path.getctime)
        log_path = latest_eval_log.split("/train.log")[0]
        print("log_path-", log_path)
        fig_.savefig(os.path.join(log_path, "train_confusion_matrix.png"))

        # Optionally, you can close the figure to free up memory
        plt.close(fig_)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        self.val_acc(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True)
    
    def predict_step(self, batch, batch_idx):
        # print(batch[0])
        # print(kwargs)
        # enable Monte Carlo Dropout
        # self.dropout.train()
        # print("batch")
        # print(batch)
        x = batch[0]
        # take average of `self.mc_iteration` iterations
        # pred = self.model(x).unsqueeze(0)
        # pred = torch.vstack(pred).mean(dim=0)
        pred = self(x)
        # print("pred")
        # print(pred)
        predicted_classes = torch.argmax(pred, dim=1)
        # print("predicted_classes", predicted_classes)
        return predicted_classes, batch[2]
        

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        preds_conf = torch.argmax(preds, dim=1)
        self.test_acc(preds, y)
        self.test_confusion(preds_conf, y)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_acc", self.test_acc, prog_bar=True, on_epoch=True)
    
    def on_test_epoch_end(self):
        print('CONFUSION_MATRIX:', self.test_confusion.compute())
        fig_, ax_ = self.test_confusion.plot()
        
        # Find the most recent metrics.csv file
        # saving to train itself may be for eval.py you can change this depending on config
        eval_log_files = glob("logs/train/multiruns/*/*/train.log")
        latest_eval_log = max(eval_log_files, key=os.path.getctime)
        log_path = latest_eval_log.split("/train.log")[0]
        # print("log_path-", log_path)
        fig_.savefig(os.path.join(log_path, "test_confusion_matrix.png"))

        # Optionally, you can close the figure to free up memory
        plt.close(fig_)


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hparams.factor,
            patience=self.hparams.patience,
            min_lr=self.hparams.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }