from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from typing import List, Dict


def train_multitask_model(model: nn.Module,
                train_loader:    DataLoader,
                val_loader:      DataLoader,
                loss_functions:  List[nn.Module],
                optimizer:       torch.optim.Optimizer,
                epochs:          int = 50,
                device:          str = "cuda" if torch.cuda.is_available() else "cpu")\
                    -> Dict:
    
    """
    loss_functions is a list of loss functions to be used for each task
    """
    model.to(device)
    best_val_loss = float("inf")
    best_model_state = None


    train_losses = []   # train loss across epochs
    val_losses   = []   # eval  loss  across epochs


    for epoch in range(epochs):
        # train loop 
        model.train()
        batch_losses = []              # train losses across batches 

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()      # Empty the gradients for each batch
            preds = model(X_batch)

            if isinstance(preds, list):
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

                    task_loss = loss_fun(p, target)
                    task_losses.append(task_loss)

                loss = sum(task_losses)     # This can turn into a weighted sum
            else:
                task_losses = [loss_functions[0](preds, y_batch)]
                loss = task_losses[0]

            loss.backward()            # This is when the magic happens
            optimizer.step()

            # batch_losses = [[t1_loss_batch1, t2_loss_batch1, ...],
            #                 [t1_loss_batch2, t2_loss_batch2, ...], ...]
            batch_losses.append([tl.detach().item() for tl in task_losses])
            

        # avg_train_losses = [avg_t1_loss, avg_t2_loss, ...]
        avg_train_losses = torch.tensor(batch_losses, requires_grad=False).mean(dim=0)

        # train_losses = [[avg_t1_loss_epoch1, avg_t2_loss_epoch1, ...],
        #                 [avg_t1_loss_epoch2, avg_t2_loss_epoch2, ...], ...]
        train_losses.append(avg_train_losses)

        # Validation loop
        model.eval()
        batch_losses = []              # eval losses across batches
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)

                if isinstance(preds, list):
                    task_losses = []
                    for i, (loss_fun, p) in enumerate(zip(loss_functions, preds)):
                        target = y_batch[:, i]
                        if target.dim() < p.dim():
                            target = target.unsqueeze(-1)

                        task_loss = loss_fun(p, target)
                        task_losses.append(task_loss)

                    loss = sum(task_losses) 
                else:
                    task_losses = [loss_functions[0](preds, y_batch)]
                    loss = task_losses[0]

                batch_losses.append([tl.item() for tl in task_losses])  # Track each task separately

        # avg_val_losses = [avg_t1_loss, avg_t2_loss, ...]
        avg_val_losses = torch.tensor(batch_losses, requires_grad=False).mean(dim=0)

        # val_losses = [[avg_t1_loss_epoch1, avg_t2_loss_epoch1, ...],
        #                 [avg_t1_loss_epoch2, avg_t2_loss_epoch2, ...], ...]
        val_losses.append(avg_val_losses)

        # Logging
        epoch_train_loss = avg_train_losses.mean().item()
        epoch_val_loss   = avg_val_losses.mean().item()


        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        task_names = [f"Task {i}" for i in range(len(loss_functions))]
        task_logs = " | ".join(
            f"{name} - Train: {tl:.4f} | Val: {vl:.4f}"
            for name, tl, vl in zip(avg_train_losses.tolist(), avg_val_losses.tolist(), task_names)
        )
        print(task_logs)

        # Save best model   
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()

    if best_model_state:
        model.load_state_dict(best_model_state)

    return {
        "model": model,
        "train_losses": torch.stack(train_losses),  # shape (epochs, n_tasks)
        "val_losses": torch.stack(val_losses)       # shape (epochs, n_tasks)
        }   


