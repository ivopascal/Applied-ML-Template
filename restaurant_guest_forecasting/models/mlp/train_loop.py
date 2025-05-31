from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from typing import List, Tuple


def train_multitask_model(model:          nn.Module,
                train_loader:   DataLoader,
                val_loader:     DataLoader,
                loss_functions: List[nn.Module],
                optimizer: torch.optim.Optimizer,
                epochs: int = 50,
                device: str = "cuda" if torch.cuda.is_available() else "cpu")\
                    -> Tuple[nn.Module, List[torch.Tensor], List[torch.Tensor]]:
    
    """
    loss_functions is a list of loss functions to be used for each task
    """
    model.to(device)
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        # train loop 
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()      # Empty the gradients for each batch
            preds = model(X_batch)

            if isinstance(preds, tuple):
                # loss_functions has a loss function for each task
                # y_batch[:, i] is the target ("Everything from the i'th column")
                # of the i'th task
                # y_batch[:, i].shape              = (batch_size,)
                # y_batch[:, i].unsqueeze(1).shape = (batch_size, 1)
                # If output is shape [batch_size, 1], we must match it with y_batch[:, i].unsqueeze(1)
                
                task_losses = []
                for i, (loss_fun, p) in enumerate(zip(loss_functions, preds)):
                    target = y_batch[:, i]
                    if target.dim() < p.dim():
                        target = target.unsqueeze(-1)

                    task_loss = loss_fun(p, target, model)
                    task_losses.append(task_loss)

                loss = sum(task_losses)
            else:
                task_losses = [loss_functions[0](preds, y_batch, model)]
                loss = task_losses[0]

            loss.backward()            # This is when the magic happens
            optimizer.step()
            train_losses.append([tl.detach().item() for tl in task_losses])

        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)

                if isinstance(preds, tuple):
                    task_losses = []
                    for i, (loss_fun, p) in enumerate(zip(loss_functions, preds)):
                        target = y_batch[:, i]
                        if target.dim() < p.dim():
                            target = target.unsqueeze(-1)

                        task_loss = loss_fun(p, target, model)
                        task_losses.append(task_loss)

                    loss = sum(task_losses) 
                else:
                    task_losses = [loss_functions[0](preds, y_batch, model)]
                    loss = task_losses[0]

                val_losses.append([tl.item() for tl in task_losses])  # Track each task separately

        avg_train_loss = torch.tensor(train_losses).mean().item()
        avg_val_loss   = torch.tensor(val_losses).mean().item()

        # Compute average per-task loss
        avg_train_losses = torch.tensor(train_losses).mean(dim=0)
        avg_val_losses   = torch.tensor(val_losses).mean(dim=0)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")  

        task_names = [f"Task {i}" for i in range(len(avg_train_losses))]
        task_logs = " | ".join(
            f"{name} - Train: {tl:.4f} | Val: {vl:.4f}"
            for name, tl, vl in zip(task_names, avg_train_losses, avg_val_losses)
        )

        print(f"Epoch {epoch+1}/{epochs} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
        print(task_logs)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


