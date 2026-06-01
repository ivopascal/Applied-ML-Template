import torch
import torchvision.models as models
import wandb
from collections.abc import Callable, Iterable
from timm.loss import AsymmetricLossMultiLabel
from typing import Literal

from chest_xray.models.train import ModelTrainer
from chest_xray.data.labels import CLASSES
from ..tools.globals import *
from ..features.evaluation import evaluate_model


class XrayClassifierBase(torch.nn.Module):
    def __init__(
            self,
            type: Literal["vgg16", "densenet"] = "vgg16",
            criterion: Callable[[tuple[torch.Tensor, torch.Tensor]],float] = AsymmetricLossMultiLabel(gamma_neg=4, gamma_pos=0, clip=0.05),
            pretrained: bool = True,
            optimizer = torch.optim.Adam,
            lr: float = 0.001,
            model: torch.nn.module | None = None
            ) -> None:
        super().__init__()
        self.type = type
        self.model = model
        self.pretrained = pretrained
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is None:
            self._build_model()
        self.optimizer = optimizer
        self.optimizer = self.optimizer(self.model.parameters(), lr=0.001)
        self.criterion = criterion
        self.modelTrainer = ModelTrainer(self.model, self.criterion, self.optimizer, self.device)


    def _build_model(self):
        if self.type == "vgg16":
            self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.model.classifier[6] = torch.nn.Linear(
                self.model.classifier[6].in_features,
                len(CLASSES)
            )
        elif self.type == "densenet":
            if self.pretrained:
                self.model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
            else:
                self.model = models.densenet161(weights=None)
            self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, len(CLASSES))
        else:
            raise ValueError("Please select a model type.")


        # take first layer
        old_first_layer = self.model.features[0]

        # Create new convolution layer with 1 input channel and keep the rest the same
        new_first_layer = torch.nn.Conv2d(
            in_channels=1,
            out_channels=old_first_layer.out_channels,
            kernel_size=old_first_layer.kernel_size,
            stride=old_first_layer.stride,
            padding=old_first_layer.padding,
        )

        # Initialize the new first layer's weights by averaging the weights of the original 3 channels
        with torch.no_grad():
            new_first_layer.weight = torch.nn.Parameter(
                old_first_layer.weight.mean(dim=1, keepdim=True)
            )
            new_first_layer.bias = old_first_layer.bias

        self.model.features[0] = new_first_layer
        self.model = self.model.to(self.device)



    def trainModel(self):
        transform = self.modelTrainer.transform_images(self.modelTrainer.image_size)
    
        for fold_idx, (train_loader, val_loader) in enumerate(
            self.modelTrainer.yield_dataloaders(transform)
        ):
            if fold_idx >= 1:
                continue
            print(f"\nFold {fold_idx + 1}/{K_FOLDS}")
    
            run = wandb.init(
                project="chest_xray",
                entity=f"chest_xray",
                name=f"fold_{fold_idx+1}: {self.type}{str(self.pretrained) if self.type == 'densenet' else ''}",
                config={
                    "fold": fold_idx + 1,
                    "epochs": 10,
                    "batch_size": BATCH_SIZE,
                    "lr": 1e-3,
                    "model": self.type,
                    "pretrained": self.pretrained
                },
            )
    
            for epoch in range(NUM_EPOCHS):
                train_loss = self.modelTrainer.train_one_epoch(
                    epoch, NUM_EPOCHS, f"Fold {fold_idx+1}", train_loader
                )
                val_loss = self.modelTrainer.validate_one_epoch(
                    epoch, NUM_EPOCHS, f"Fold {fold_idx+1}", val_loader
                )
    
                # log to wandb
                run.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })
    
                print(
                    f"Fold {fold_idx+1} Epoch {epoch+1}: "
                    f"Train {train_loss:.4f} | Val {val_loss:.4f}"
                )
                path: str = f"{MODEL_PATH}{self.type}_{"pretrained" if self.pretrained else "scratch"}_epoch{epoch}.pth"
                torch.save(self.model, path)
                print(f"model saved: {path}")
            run.finish()
            self.evaluate()

    
    def evaluate(self):
        eval_loader = None
        classes = [c for c in CLASSES]
        transform = self.modelTrainer.transform_images(self.modelTrainer.image_size)
        for i, loaders in enumerate(self.modelTrainer.yield_dataloaders(transform)):
            if i > 0:
                continue
            eval_loader = loaders[1]    # takes validation loader, so model only tests on data which it hasn't trained on
        results_df, summary, confusion_matrices = evaluate_model(
            self.model,
            eval_loader,
            self.device,
            classes
        )
        print("summary:")
        for key, val in summary.items():
            print(f"{key}: {val}")
        print("\n\ndataframe:")
        print(results_df)
        print("\n\nconfusion matrices:")
        print(confusion_matrices)