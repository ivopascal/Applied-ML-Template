import os
import pickle

import pandas as pd

from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from restaurant_guest_forecasting.models.losses.asymmetric_loss \
    import AsymmetricL2MSE

from restaurant_guest_forecasting.models.mlp.mlp import MultiTaskMLP
from restaurant_guest_forecasting.models.mlp.train_loop import train_multitask_model

from restaurant_guest_forecasting.data.train_test_split import train_val_test_data
from sklearn.model_selection import KFold
from restaurant_guest_forecasting.data.tensor_data import prepare_dataloader,\
                                                    guest_df_to_tensor_dataset

from restaurant_guest_forecasting.models.utils.plotting import \
                                    plot_avg_loss_over_epochs

PARAM_GRID = {
        'dropout_rate': [0.0, 0.1, 0.2],
        'num_layers': [2, 4, 6],
        'neurons_per_layer': [37, 64, 128],
        'l2_lambda': [0.0, 1e-2, 1e-1, 1.0]
    }

NORMLIZE = True  # Set to True if you want to normalize the data


def train_save_mlp_guests(model_file_name: str = "guests_mlp.pt"):
     
    train_df, val_df, test_df = train_val_test_data()

    # keep only a subset of the training data for forcing overfitting
    # train_df = train_df.sample(n=1, random_state=42).reset_index(drop=True)

    # Prepare DataLoaders for guests only
    # Training DataLoader
    train_loader = prepare_dataloader(df=train_df,
                                      batch_size=64, 
                                      to_tensor_fn=guest_df_to_tensor_dataset,
                                      is_train=True,
                                      normalize=NORMLIZE)

    # Get input size from a batch
    sample_batch = next(iter(train_loader))
    X_sample, _ = sample_batch
    input_size = X_sample.shape[1]

    # # Forcing overfitting by using a fake dataset
    # # This is just to test the training loop and the model
    # X_fake = torch.randn(100, input_size)
    # y_fake = torch.linspace(0, 1, 100).unsqueeze(1)    # unique targets
    # ds_fake = torch.utils.data.TensorDataset(X_fake, y_fake)
    # train_loader  = DataLoader(ds_fake, batch_size=10, shuffle=False)

    # Validation DataLoader
    val_loader   = prepare_dataloader(df=val_df,
                                      batch_size=1,
                                      to_tensor_fn=guest_df_to_tensor_dataset,
                                      is_train=False,
                                      normalize=NORMLIZE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # neurons = [input_size, input_size, input_size]
    neurons = [input_size] + [input_size]*6

    print(f"Input size: {input_size}")
    print(f"Neurons per layer: {neurons}")

    # One Task single value regression
    output_neurons = [1]

    # Build model
    single_task_mlp = MultiTaskMLP(num_neurons=neurons, 
                                   droput_rate=0.0, 
                                   activation="relu", 
                                   output_neurons=output_neurons)
    
    single_task_mlp = single_task_mlp.to(device=device)
    
    # Define Loss Function
    # Penalize 1.5 as harshly overestimation
    w_over = 2
    w_under = 1
    loss_fn = AsymmetricL2MSE(w_over=w_over,
                              w_under=w_under, 
                              model=single_task_mlp,
                              l2_lambda=0.0).to(device=device)
    # Only plain MSE
    # loss_fn = torch.nn.MSELoss().to(device=device)

    # Single loss
    losses = [loss_fn]

    # Define optimizer
    optimizer = optim.Adam(single_task_mlp.parameters(), lr=1e-4)
    # optimizer = optim.SGD(single_task_mlp.parameters(),
    #                       lr=1e-4,
    #                       momentum=0.9,
    #                       weight_decay=1e-4)

    # Epochs
    epochs = 1000

    # Train the model
    train_info = train_multitask_model(
        model=single_task_mlp,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_functions=losses,
        optimizer=optimizer,
        epochs=epochs
    )

    trained_model = train_info["model"]
    train_losses  = train_info["train_losses"]
    val_losses    = train_info["val_losses"]

    print(f"{train_losses=}\n{val_losses=}")

    plot_avg_loss_over_epochs(train_losses, val_losses)

    # Define the save path
    save_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, model_file_name)
    torch.save(trained_model.state_dict(), save_path)

    print(f"Model saved to {save_path}")

def tune_hyperparameters():

    train_df, val_df, test_df = train_val_test_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare validation DataLoader
    val_loader = prepare_dataloader(df=val_df,
                                    batch_size=1,
                                    to_tensor_fn=guest_df_to_tensor_dataset,
                                    is_train=False)

    param_grid = PARAM_GRID


    all_configs = list(product(
        param_grid['dropout_rate'],
        param_grid['num_layers'],
        param_grid['neurons_per_layer'],
        param_grid['l2_lambda']
    ))

    best_val_loss = float('inf')
    best_params = None
    best_model = None
    results_dict = {}

    for dropout, num_layers, neurons, l2_lambda in all_configs:
        print(f"\nTrying: dropout={dropout}, layers={num_layers}, neurons={neurons}, l2_lambda={l2_lambda}")

        # Prepare training DataLoader
        train_loader = prepare_dataloader(df=train_df,
                                          batch_size=64,
                                          to_tensor_fn=guest_df_to_tensor_dataset,
                                          is_train=True)

        # Get input size
        X_sample, _ = next(iter(train_loader))
        input_size = X_sample.shape[1]

        # Define architecture
        neurons_list = [input_size] + [neurons] * num_layers

        # Build model
        model = MultiTaskMLP(num_neurons=neurons_list,
                             droput_rate=dropout,
                             activation="relu",
                             output_neurons=[1]).to(device)

        # Use AsymmetricL2MSE loss for consistency
        loss_fn = AsymmetricL2MSE(w_over=2.0, w_under=1.0, model=model, l2_lambda=l2_lambda).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Train model
        train_info = train_multitask_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_functions=[loss_fn],
            optimizer=optimizer,
            epochs=50  # Reduced epochs for faster tuning
        )

        final_val_loss = train_info["val_losses"][-1][0]

        config_tuple = (dropout, num_layers, neurons, l2_lambda)
        results_dict[config_tuple] = final_val_loss

        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_params = {
                'dropout_rate': dropout,
                'num_layers': num_layers,
                'neurons_per_layer': neurons,
                'l2_lambda': l2_lambda
            }
            best_model = model

        print(f"Validation loss: {final_val_loss:.4f}")

    print("\nBest parameters:", best_params)
    print("Best validation loss:", best_val_loss)

    # Serialize results
    save_dir = os.path.join(os.path.dirname(__file__), "saved_models", "hyperparameters_search")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "hyperparam_search_results.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"Hyperparameter search results saved to {save_path}")

    return best_model, best_params

def k_fold_cross_validation_guests(k=5, epochs=100):
    train_df, val_df, test_df = train_val_test_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Combine train and val for cross-validation
    full_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_df)):
        print(f"\nFold {fold+1}/{k}")

        fold_train_df = full_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = full_df.iloc[val_idx].reset_index(drop=True)

        train_loader = prepare_dataloader(df=fold_train_df,
                                          batch_size=16,
                                          to_tensor_fn=guest_df_to_tensor_dataset,
                                          is_train=True)
        val_loader = prepare_dataloader(df=fold_val_df,
                                        batch_size=16,
                                        to_tensor_fn=guest_df_to_tensor_dataset,
                                        is_train=False)

        X_sample, _ = next(iter(train_loader))
        input_size = X_sample.shape[1]
        neurons = [input_size, 64, 64, 32]
        output_neurons = [1]

        model = MultiTaskMLP(num_neurons=neurons,
                             droput_rate=0.0,
                             activation="relu",
                             output_neurons=output_neurons).to(device)

        # Use the same asymmetric loss as in train_save_mlp_guests
        w_over = 2.0
        w_under = 1.0
        loss_fn = AsymmetricL2MSE(w_over=w_over,
                                  w_under=w_under,
                                  model=model,
                                  l2_lambda=0.0).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-2)

        train_info = train_multitask_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_functions=[loss_fn],
            optimizer=optimizer,
            epochs=epochs
        )

        final_val_loss = train_info["val_losses"][-1][0]
        print(f"Fold {fold+1} validation loss: {final_val_loss:.4f}")
        fold_results.append(final_val_loss)

    avg_val_loss = sum(fold_results) / len(fold_results)
    print(f"\nAverage validation loss across {k} folds: {avg_val_loss:.4f}")
    return fold_results



def k_fold_cross_validation_with_hyperparam_search(k=5, epochs=50):
    
    train_df, val_df, _ = train_val_test_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Combine training + validation for cross-validation
    full_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    kf = KFold(n_splits=k, shuffle=False)

    param_grid = PARAM_GRID

    all_configs = list(product(
        param_grid['dropout_rate'],
        param_grid['num_layers'],
        param_grid['neurons_per_layer'],
        param_grid['l2_lambda']
    ))

    results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_df)):
        print(f"\n=== Fold {fold + 1}/{k} ===")

        fold_train_df = full_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df   = full_df.iloc[val_idx].reset_index(drop=True)

        best_loss = float('inf')
        best_config = None
        best_model_state = None

        for dropout, num_layers, neurons, l2_lambda in all_configs:
            print(f"Trying config: dropout={dropout}, layers={num_layers}, neurons={neurons}, l2_lambda={l2_lambda}")

            train_loader = prepare_dataloader(fold_train_df, batch_size=16,
                                              to_tensor_fn=guest_df_to_tensor_dataset, is_train=True)
            val_loader = prepare_dataloader(fold_val_df, batch_size=16,
                                            to_tensor_fn=guest_df_to_tensor_dataset, is_train=False)

            X_sample, _ = next(iter(train_loader))
            input_size = X_sample.shape[1]
            architecture = [input_size] + [neurons] * num_layers

            model = MultiTaskMLP(num_neurons=architecture,
                                 droput_rate=dropout,
                                 activation="relu",
                                 output_neurons=[1]).to(device)

            loss_fn = AsymmetricL2MSE(w_over=2.0, w_under=1.0, model=model, l2_lambda=l2_lambda).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-2)  # fixed learning rate, no weight_decay

            train_info = train_multitask_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_functions=[loss_fn],
                optimizer=optimizer,
                epochs=epochs
            )

            val_loss = train_info["val_losses"][-1][0]
            print(f"Validation loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_config = {
                    'dropout_rate': dropout,
                    'num_layers': num_layers,
                    'neurons_per_layer': neurons,
                    'l2_lambda': l2_lambda
                }
                best_model_state = model.state_dict()

        print(f"\nBest config for fold {fold+1}: {best_config}")
        results.append((best_loss, best_config))

    return results


def main():
    # tune_hyperparameters()
    train_save_mlp_guests(model_file_name="guests_mlp.pt")
    

if __name__ == "__main__":
    main()