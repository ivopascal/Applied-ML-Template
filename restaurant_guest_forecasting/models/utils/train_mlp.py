import os

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from restaurant_guest_forecasting.models.losses.asymmetric_loss \
    import AsymmetricL2MSE

from restaurant_guest_forecasting.models.mlp.mlp import MultiTaskMLP
from restaurant_guest_forecasting.models.mlp.train_loop import train_multitask_model

from restaurant_guest_forecasting.data.train_test_split import train_val_test_data
from restaurant_guest_forecasting.data.tensor_data import prepare_dataloader,\
                                                    guest_df_to_tensor_dataset




def train_save_mlp_guests(model_file_name: str = "guests_mlp.pt"):
     
    train_df, val_df, test_df = train_val_test_data()

    # Prepare DataLoaders for guests only
    train_loader = prepare_dataloader(df=train_df,
                                      batch_size=64, 
                                      to_tensor_fn=guest_df_to_tensor_dataset)
    val_loader   = prepare_dataloader(df=val_df,
                                      batch_size=64,
                                      to_tensor_fn=guest_df_to_tensor_dataset)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get input size from a batch
    sample_batch = next(iter(train_loader))
    X_sample, _ = sample_batch
    input_size = X_sample.shape[1]

    # Two hidden layers all with `input_size` neurons
    neurons = [input_size, input_size, input_size]
    # One Task single value regression
    output_neurons = [1]

    # Build model
    single_task_mlp = MultiTaskMLP(num_neurons=neurons, 
                                   droput_rate=0.0, 
                                   activation="relu", 
                                   output_neurons=output_neurons)
    
    single_task_mlp = single_task_mlp.to(device=device)
    
    # Define Loss Function
    # Penalize twice as harshly overestimation
    w_over = 2.0
    w_under = 1.0
    loss_fn = AsymmetricL2MSE(w_over=w_over,
                              w_under=w_under, 
                              model=single_task_mlp).to(device=device)
    # Single loss
    losses = [loss_fn]

    # Define optimizer
    optimizer = optim.Adam(single_task_mlp.parameters(), lr=1e-3)

    # Epochs
    epochs = 50

    # Train the model
    trained_model, train_losses, val_losses = train_multitask_model(
        model=single_task_mlp,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_functions=losses,
        optimizer=optimizer,
        epochs=epochs
    )

    # Define the save path
    save_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, model_file_name)
    torch.save(trained_model.state_dict(), save_path)

    print(f"Model saved to {save_path}")

def main():
    train_save_mlp_guests()


if __name__ == "__main__":
    main()