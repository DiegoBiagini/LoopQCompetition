import string
from typing import Callable
from .model import CNNATT
import torch
from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
from .data import input_pipeline_train, batch_pad_collate_fn

BATCH_SIZE = 32

def init_model(n_classes : int):
    """Initialize everything that is needed to train a model

    Args:
        n_classes (int): Number of classes the model will choose from

    Returns:
        dict: Dictionary containing: 
            "model": model object, 
            "train_function": partial function to call to train the model, model and input_pipeline already given
    """
    model = CNNATT(n_classes)
    return {
        "model" : model,
        "train_function" : partial(train_model, model=model, input_pipeline=input_pipeline_train)
    }

def load_train_state(path : Path):
    """Load a model from file

    Args:
        path (Path): Path of the .tar file containing train state information

    Returns:
        dict: Dictionary containing:
            "model": loaded model object,
            "train_function": partial function to train the model, 
            where the model, input pipeline, starting epoch and optimizer state are already set
        list: List of dicts containing the training history (train/val loss/acc and epoch)
    """
    checkpoint = torch.load(path)
    model = CNNATT(checkpoint["n_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded train state for CNN_ATT, last epoch:", checkpoint["epoch"])
    return {
        "model" : model,
        "train_function" : partial(train_model, model=model, input_pipeline = input_pipeline_train,
                            starting_epoch = checkpoint["epoch"]+1, optimizer_state_dict = checkpoint["optimizer_state_dict"],
                            train_history = checkpoint["history"])
    }, checkpoint["history"]


def train_model(model : CNNATT , train_ds : torch.utils.data.Dataset,
                    input_pipeline : Callable, device : string,
                    val_ds : torch.utils.data.Dataset = None,
                    train_checkpointing : int = 0, save_base_folder : Path = None,
                    starting_epoch : int = 1, optimizer_state_dict : dict = None, train_history : list = None):
    """Function to train the model

    Args:
        model (CNNATT): model
        train_ds (torch.utils.data.Dataset): train dataset
        input_pipeline (Callable): function applied to the batch before training
        device (string): training device
        val_ds (torch.utils.data.Dataset, optional): validation dataset
        train_checkpointing (int, optional): Number of epochs between each checkpoint, if 0 do not checkpoint. Defaults to 0.
        save_base_folder (Path, optional): Folder where to put the checkpoint. Defaults to None.
        starting_epoch (int, optional): Starting epoch, used to resume training. Defaults to 1.
        optimizer_state_dict (dict, optional): State to initialize the optimizer to, used to resume training. Defaults to None.
        train_history (list, optional): Train history to initialize the training to. Defaults to None
    """

    model = model.to(device)
    model.train()
    print("Training CNNATT model")
    
    if train_checkpointing != 0:
        assert save_base_folder != None, "Train checkpointing was requested but no folder was provided"
        model_save_folder = save_base_folder / "CNNATT_train"
        model_save_folder.mkdir(exist_ok=True)
        base_save_name = ( model_save_folder / "CNNATT_epoch_").resolve()
    
    n_epochs = 300
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Optimizer, if a state dict was passed initialize it with those parameters
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=1e-6)
    
    if optimizer_state_dict != None:
        optimizer.load_state_dict(optimizer_state_dict)

    # Decay lr by 10 every 20 epochs with no improvement
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=20, factor=0.1,verbose=True, 
        min_lr=1e-5)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1, verbose=True)

    # Keep track of training history as well
    if train_history is None:
        history = []
    else: 
        history = train_history
    
    for epoch in range(starting_epoch, n_epochs+1):
        print("Epoch:", epoch)

        batched_ds = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers = 2, pin_memory=True,
            collate_fn=batch_pad_collate_fn)
        
        model.train()
        epoch_losses = []
        epoch_accuracies = []
        for batch, batch_data in enumerate(batched_ds):

            X = input_pipeline(batch_data["soundwave"]).to(device)
            y = batch_data["emotion"].to(device)

            # Forward and backward pass
            optimizer.zero_grad()

            logits = model(X)
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()
            
            accuracy = torch.mean((torch.argmax(torch.softmax(logits, dim = 1), dim=1) == y).type(torch.float32))
            loss = loss.item()
            
            epoch_losses += [loss]
            epoch_accuracies += [accuracy]
            
            if batch % 50 == 0:
                current = batch * X.shape[0]

                print(f"loss: {loss:>7f} [{current:>5d}/{len(train_ds):>5d}]")
        
        train_loss = torch.mean(torch.as_tensor(epoch_losses))
        train_acc = torch.mean(torch.as_tensor(epoch_accuracies))
        print(f"Train loss:{train_loss:>7f}, Train accuracy: {train_acc:>4f}")

        # If a validation dataset is provided use it
        val_loss = None
        val_acc = None
        if val_ds != None:
            batched_val_ds = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers = 2, pin_memory=True,
                collate_fn=batch_pad_collate_fn)
            
            model.eval()
            val_losses = []
            val_accuracies = []
            for batch, batch_data in enumerate(batched_val_ds):

                X = input_pipeline(batch_data["soundwave"]).to(device)
                y = batch_data["emotion"].to(device)

                logits = model(X)
                loss = loss_fn(logits, y)

                accuracy = torch.mean((torch.argmax(torch.softmax(logits, dim = 1), dim=1) == y).type(torch.float32))
                loss = loss.item()
                
                val_losses += [loss]
                val_accuracies += [accuracy]

            val_loss = torch.mean(torch.as_tensor(val_losses))
            val_acc = torch.mean(torch.as_tensor(val_accuracies))
            print(f"Val loss:{val_loss:>7f}, Val accuracy: {val_acc:>4f}")

        epoch_metrics = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                             "val_loss": val_loss, "val_acc": val_acc}
        history += [epoch_metrics]
        
        # Send val loss information to scheduler
        scheduler.step(val_loss)
        #scheduler.step()
        
        # Save train state (model params, optimizer state, epoch)
        if train_checkpointing != 0:
            if epoch % train_checkpointing == 0:
                epoch_train_state_path = (str(base_save_name) + str(epoch) + ".tar")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'n_classes' : model.n_classes,

                    'history': history
                    }, epoch_train_state_path)
                print("Checkpointed train state in :", epoch_train_state_path)


        
