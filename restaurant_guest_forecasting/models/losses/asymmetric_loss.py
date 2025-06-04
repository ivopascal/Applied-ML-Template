import torch
import torch.nn as nn

class AsymmetricL2MSE(nn.Module):
    """
    Asymmetric Mean Squared Error (L2) loss with optional L2 regularization.

    This loss function penalizes overestimations and underestimations with 
    different weights, making it ideal for applications like demand forecasting 
    where overprediction may be more costly than underprediction.

    Attributes:
        w_over (float): Weight for overestimation errors (pred > target).
        w_under (float): Weight for underestimation errors (pred <= target).
        l2_lambda (float): L2 regularization strength.
    """

    def __init__(self, w_over: float = 2.0,
                 w_under: float = 1.0,
                 l2_lambda: float = 0.0, 
                 model: nn.Module | None = None) -> None:
        """
        Initializes the AsymmetricL2MSE loss.

        Args:
            w_over (float): Multiplier for squared errors when preds > targets.
            w_under (float): Multiplier for squared errors when preds <= targets.
            l2_lambda (float): Strength of L2 regularization on model parameters.
        """
        super().__init__()
        self._w_over    = w_over       # Overestimation weights 
        self._w_under   = w_under      # Underestimation weights 
        self._l2_lambda = l2_lambda    # Regularization term
        self._model      = model        # Save model regularizing weights      

    def forward(
            self,
            preds: torch.Tensor, 
            targets: torch.Tensor,
            ) -> torch.Tensor:
        """
        Computes the asymmetric MSE loss, optionally including L2 regularization.

        Args:
            preds (torch.Tensor): Predicted values from the model. Shape: (batch_size,).
            targets (torch.Tensor): Ground truth target values. Shape: (batch_size,).
            model (torch.nn.Module, optional): Model used for training. If provided and 
                `l2_lambda` > 0, L2 regularization is applied to its parameters.

        Returns:
            torch.Tensor: A scalar tensor representing the total loss.
        """
        sq_error = (preds - targets).pow(2)

        # Choose weight per element
        weights = torch.where(preds > targets,
                              self._w_over,
                              self._w_under)
        
        # Compute MSE
        loss = (weights * sq_error).mean()

        if self._model is not None and self._l2_lambda > 0.0:
            l2_term = sum(p.pow(2).sum() for p in self._model.parameters())
            loss += self._l2_lambda * l2_term

        return loss