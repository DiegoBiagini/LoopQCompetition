import pandas as pd
from pathlib import Path
import numpy as np
import shutil
from torch.utils.data import Dataset
import torch
import soundfile as sf
import librosa
from clint.textui import progress
import requests

# Static mapping so there is no confusion
emotion_to_ordinal_dict = {"angry":0, "disgust":1, "fear":2, "happy":3, "neutral":4, "sadness":5, "surprise":6}
ordinal_to_emotion_dict = {0: "angry", 1: "disgust", 2:"fear", 3:"happy", 4:"neutral", 5:"sadness", 6:"surprise"}


class SERDataset(Dataset):

    def __init__(self, dataset_file : Path, ds_dir : Path, base_sample_rate : int =None, data_transform=None):
        """
        Args:
            ds_file (Path): Path to the csv file with filename-emotions.
            ds_dir (Path): Directory with all the audio files.
            base_sample_rate(int): Sample rate to force files to be equal to, if a file with a wrong sample rate is found throw an error
            transform (callable, optional): Optional transform to be applied on a sample.

        """
        self.ds = pd.read_csv(dataset_file)
        self.ds_dir = ds_dir
        self.data_transform = data_transform
        self.base_sample_rate = base_sample_rate
        
        # Remove null values and map emotion to integer
        self.ds = self.ds[~self.ds.isnull().any(axis=1)]
        self.ds.reset_index(drop=True, inplace=True)

        self.ds["emotion"] = self.ds["emotion"].map(emotion_to_ordinal_dict)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = (self.ds_dir / self.ds["file_id"][idx]).resolve()
        
        # Load at a certain sr using librosa, torchaudio is extremely slow at resampling
        ts, sr = librosa.load(filename, sr=self.base_sample_rate)

        ts = torch.as_tensor(ts)
        if len(ts.shape) <2:
            ts = torch.unsqueeze(ts, 0)

        # Some files might have multiple tracks, throw away everything except the first one
        # Honestly you could keep it as one more training example 
        if ts.shape[0] != 1:
            ts = ts[:1,:]           # Take first element while keeping (1,...) dimension

        if self.data_transform:
            ts = self.data_transform(ts)

        sample = {"soundwave": torch.as_tensor(ts), "emotion":self.ds["emotion"][idx]}
        
        return sample


def split_labeled_dataset(trainset_info_path : Path, folder_destination_path : Path, train_split : float, resample : int = -1):
    """Split the labeled dataset(also known as the challenge train dataset) into train and test set
    Mantain the same ratio of samples from each origin as the full training dataset
    Two new csv files will be created at folder_destination_path, as well as two new folders in 
    which audio files of the new train/test set will be copied

    Args:
        trainset_info_path (Path): Path of the csv files containing the train dataset information
        folder_destination_path (Path): Path of where to save the new train/test datasets
        train_split (float): Percentage of samples used for training
        resample (int, optional): Whether to resample to target sampling rate, do not resample if it's -1
    """
    # Load original file and create train and test dataframes
    train_ds = pd.read_csv(trainset_info_path, index_col=0)
    # Drop columns with any null values
    train_ds = train_ds[~train_ds.isnull().any(axis=1)]

    new_train_ds = pd.DataFrame(columns = train_ds.columns)
    new_test_ds = pd.DataFrame(columns = train_ds.columns)

    # To keep ratio of dataset origin group by origin and
    # add the chosen proportion to the train/test ds
    for g_name, group in train_ds.groupby("origin"):
        group_size = group.shape[0]
        group_split_size = int(train_split * group_size)

        train_chosen_samples = list(np.random.choice(group_size, size=group_split_size, replace=False))
        test_chosen_samples = list(set(range(group_size)) - set(train_chosen_samples))

        new_train_ds = pd.concat([new_train_ds, group.iloc[train_chosen_samples,:]])
        new_test_ds = pd.concat([new_test_ds, group.iloc[test_chosen_samples,:]])

    new_train_filename = (folder_destination_path / trainset_info_path.name)
    new_test_filename = (folder_destination_path / str(trainset_info_path.name).replace("train", "test"))
    
    # Save the new csv files
    new_train_ds.to_csv(new_train_filename)
    new_test_ds.to_csv(new_test_filename)
    print("Created new train/test dataset files:", new_train_filename, " ", new_test_filename)

    # Create new train/test folders
    new_train_folder = (folder_destination_path / "train")
    new_test_folder = (folder_destination_path / "test")

    new_train_folder.mkdir(exist_ok=True)
    new_test_folder.mkdir(exist_ok=True)
    # Delete anything that was there
    [f.unlink() for f in new_train_folder.glob("*") if f.is_file()] 
    [f.unlink() for f in new_test_folder.glob("*") if f.is_file()] 


    # Copy the chosen files in the appropriate folder
    print("Copying audio files to new locations")

    for _,row in new_train_ds.iterrows():
        src_path = (trainset_info_path.parent / "train" / row['file_id'])
        dst_path = (new_train_folder / row['file_id'])
        
        if resample == -1:
            shutil.copy(src_path, dst_path)
        else:
            y, sr = librosa.load(src_path)
            if sr != resample:
                y = librosa.resample(y, orig_sr=sr, target_sr=resample)
            sf.write(dst_path, y, resample, 'PCM_24')

    for _,row in new_test_ds.iterrows():
        src_path = (trainset_info_path.parent / "train" / row['file_id'])
        dst_path = (new_test_folder / row['file_id'])
        
        if resample == -1:
            shutil.copy(src_path, dst_path)
        else:
            y, sr = librosa.load(src_path)
            if sr != resample:
                y = librosa.resample(y, orig_sr=sr, target_sr=resample)
            sf.write(dst_path, y, resample, 'PCM_24')
            
    print("Files copied to new locations:", new_train_folder, " ", new_test_folder )


# Download a file 
def download_file(url, local_filename):
    chunk_size = 8096
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=chunk_size), expected_size=(total_length/chunk_size) + 1): 
                if chunk:
                    f.write(chunk)
                    f.flush()

    return local_filename