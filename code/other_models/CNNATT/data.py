import torch
import torchaudio
from pathlib import Path
import librosa
from torchvision import transforms
import numpy as np
from .model import CNNATT
from torch.nn.utils.rnn import pad_sequence

# Constants for this model
# Base sample rate all the files will respect
SAMPLE_RATE = 16000
# Duration an audio clip must respect (s)
CLIP_LENGTH = 5

WINDOW_FRAMES = 1024
HOP_FRAMES = 256 
N_MELS = 128
N_MFCC = 40

def input_pipeline_train(batch_waveform : torch.Tensor):
    # Obtain MFCCs from waveform

    mfcc = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=40, 
                melkwargs={
                    "n_fft": WINDOW_FRAMES,
                    "n_mels": N_MELS,
                    "hop_length": HOP_FRAMES,
                    "mel_scale": "htk"})
    return mfcc(batch_waveform)
    

def input_pipeline_inference(batch_waveform : torch.Tensor):
    # Obtain MFCCs from waveform

    mfcc = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=40, 
                melkwargs={
                    "n_fft": WINDOW_FRAMES,
                    "n_mels": N_MELS,
                    "hop_length": HOP_FRAMES,
                    "mel_scale": "htk"})
    return mfcc(batch_waveform)


def init_train_dataset(csv_file_path : Path, audio_folder_path : Path, dataset_class):
    # Randomly speedup/slowdown audio
    rate_choice = np.random.uniform(0.9,1.1)
    time_stretch_f = lambda y : librosa.effects.time_stretch(y.numpy(), rate = rate_choice)

    # Trim trailing silence
    trim_silence_f = lambda y : librosa.effects.trim(y, top_db=20)[0]

    # Reshape the soundwave to be exactly CLIP_LENGTH seconds long
    trg_size = SAMPLE_RATE*CLIP_LENGTH
    cut_audio_f = lambda y : y[:,:trg_size] if y.shape[1] >= trg_size else np.pad(y, ((0,0),(0,trg_size-y.shape[1])))
    
    t_dataset = dataset_class(csv_file_path, audio_folder_path, base_sample_rate=SAMPLE_RATE, 
                            data_transform= transforms.Compose([
                                # transforms.Lambda(time_stretch_f),
                                transforms.Lambda(trim_silence_f), 
                                #transforms.Lambda(cut_audio_f)  
                            ]))
    return t_dataset


def init_test_dataset(csv_file_path : Path, audio_folder_path : Path, dataset_class):
    # Trim trailing silence
    trim_silence_f = lambda y : librosa.effects.trim(y, top_db=20)[0]

    # Reshape the soundwave to be exactly CLIP_LENGTH seconds long
    trg_size = SAMPLE_RATE*CLIP_LENGTH

    cut_audio_f = lambda y : y[:,:trg_size] if y.shape[1] >= trg_size else np.pad(y, ((0,0),(0,trg_size-y.shape[1])))
    
    t_dataset = dataset_class(csv_file_path, audio_folder_path, base_sample_rate=SAMPLE_RATE, 
                            data_transform= transforms.Compose([
                                transforms.Lambda(trim_silence_f), 
                                #transforms.Lambda(cut_audio_f)  
                            ]))
    return t_dataset


def load_inference_model(path : Path):
    """Load a model for inference from a train checkpoint

    Args:
        path (Path): Path to the .tar training checkpoint

    Returns:
        dict: Dict containing:
            "model":the model ready for inference (it outputs probabilities instead of logits)
            "inference_pipeline" : pipeline through which to pass batches of raw data before training
        list: List of dicts containing the training history (train/val loss/acc and epoch)
    """
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else: 
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model = CNNATT(checkpoint["n_classes"], inference=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "model" : model,
        "inference_pipeline" : input_pipeline_inference
    }, checkpoint["history"]


def batch_pad_collate_fn(data):
    soundwave_list = [torch.squeeze(el["soundwave"]) for el in data]
    emotion_list = [torch.as_tensor(el["emotion"]).view(1,-1) for el in data]

    padded_soundwaves = torch.unsqueeze(pad_sequence(soundwave_list, batch_first=True),1)
    concat_emotions = torch.squeeze(torch.cat(emotion_list, 0))

    return {"soundwave":padded_soundwaves, "emotion":concat_emotions}