import torch
import torch.nn as nn

class AsymmetricL2MSE(nn.Module):
    
    def __init__(self, w_over: float = 2.0, w_under: float = 1.0,
                  l2_lambda: float = 0.0,) -> None:
        super().__init__()
        self._w_over    = w_over       # Overestimation weights 
        self._w_under   = w_under      # Underestimation weights 
        self._l2_lambda = l2_lambda    # Regularization term 

    def forward(
            self,
            preds: torch.Tensor, 
            targets: torch.Tensor,
            model: torch.Module | None = None
            ) -> torch.Tensor:
        
        sq_error = (preds - targets).pow(2)

        # Choose weight per element
        weights = torch.where(preds > targets,
                              self._w_over,
                              self._w_under)
        
        # Compute MSE
        loss = (weights * sq_error).mean()

        if model is not None and self._l2_lambda > 0.0:
            l2_term = sum(p.pow(2).sum() for p in model.parameters())
            loss += self._l2_lambda * l2_term

        return loss