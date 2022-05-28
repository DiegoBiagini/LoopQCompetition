from data_utils import *
from MMFUSION.train import init_model, load_train_state
from MMFUSION.data import init_train_dataset
from data_utils import SERDataset
import sys

mod_path = Path(__file__).parent

def train(csv_path, audio_path):
    # Choosing device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    # Initialize datasets
    t_dataset = init_train_dataset(csv_path, audio_path, SERDataset)
    print(f"Initalized dataset at:\n {csv_path} \n{audio_path}")

    # Where to save model while training
    saved_model_path = (mod_path / "saved_models")

    # Choose model
    model_collation = init_model(n_classes = 7)
    train_function = model_collation["train_function"]

    train_function(train_ds=t_dataset, device=device, train_checkpointing = True, save_base_folder = saved_model_path)


def resume_train(csv_path, audio_path, resume_file_path):
    # Choosing device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    #Initialize dataset    
    t_dataset = init_train_dataset(csv_path, audio_path, SERDataset)
    print(f"Initalized dataset at:\n{csv_path} \n{audio_path}")

    # Where to save model while training
    saved_model_path = (mod_path / "saved_models")
   
    # Choose file from where to resume training
    model_collation, previous_history = load_train_state(resume_file_path)

    print("Previous history")
    for e in previous_history:
        print(e)

    train_function = model_collation["train_function"]
    train_function(train_ds = t_dataset, device=device, train_checkpointing=True, save_base_folder = saved_model_path)

"""
Usage

Start training from scratch:
python training.py start csv_path audio_path

Resume training:
python training.py resume csv_path audio_path resume_weights_path

csv_path : file to classify
audio_path : folder containing the audio files whose names are in the csv file
resume_weights_path : path where the temporary weights are saved, optional
"""
if __name__ == "__main__":
    """
    Default values:

    model_weights_path = mod_path / "saved_models/MMFUSION_train/MMFUSION.tar"
    csv_path = mod_path / "datasets/challengeA_data/2022challengeA_train.csv"
    audio_path = mod_path / "datasets/challengeA_data/train"
    mode = sys.argv[1]
    """
    
    assert len(sys.argv) >= 4, "At least 3 parameters are needed: <mode> <csv_path> <audio_path>"
    mode = sys.argv[1]
    assert mode == "start" or mode == "resume", "Mode has to be either 'start' or 'resume'" 

    if mode == "start":
        assert len(sys.argv) == 4, "With 'start' mode the parameters have to be 3: start <csv_path> <audio_path>"
    else:
        assert len(sys.argv) <=5, "With 'resume' mode the parameters have to be at least 3 and at most 4: resume <csv_path> <audio_path> [resume_weights_path]"

    csv_path = Path(sys.argv[2])
    audio_path = Path(sys.argv[3])

    if mode == "resume":
        if len(sys.argv < 5):
            model_weights_path = "/saved_models/MMFUSION_train/MMFUSION.tar"
        else:
            model_weights_path = Path(sys.argv[4])

    if mode=="start":
        print("Starting training from zero")
        train(csv_path, audio_path)
    else:
        print("Resuming training")
        resume_train(csv_path, audio_path, model_weights_path)
