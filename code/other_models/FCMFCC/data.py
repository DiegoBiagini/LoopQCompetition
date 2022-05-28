import torch
import torchaudio
from pathlib import Path
import librosa
from torchvision import transforms
import numpy as np
from .model import FC_MFCC

# Constants for this model
# Base sample rate all the files will respect
SAMPLE_RATE = 16000
# Duration an audio clip must respect (s)
CLIP_LENGTH = 5

def input_pipeline_train(batch_waveform : torch.Tensor):
    # Obtain MFCCs from waveform
    window_frames = 1024 #64ms window -> 16 frames/ms -> 64*16 frames per window
    hop_frames = 256 # 16ms overlap -> 16*16 frames overlap
    mfcc = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=40, 
                melkwargs={
                    "n_fft": window_frames,
                    "n_mels": 128,
                    "hop_length": hop_frames,
                    "mel_scale": "htk"})
    return mfcc(batch_waveform)
    

def input_pipeline_inference(batch_waveform : torch.Tensor):
    window_frames = 1024 #64ms window -> 16 frames/ms -> 64*16 frames per window
    hop_frames = 256 # 16ms overlap -> 16*16 frames overlap
    mfcc = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=40, 
                melkwargs={
                    "n_fft": window_frames,
                    "n_mels": 128,
                    "hop_length": hop_frames,
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
                                transforms.Lambda(cut_audio_f)  ]))
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
                                transforms.Lambda(cut_audio_f)  ]))
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
    model = FC_MFCC(checkpoint["n_classes"], inference=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "model" : model,
        "inference_pipeline" : input_pipeline_inference
    }, checkpoint["history"]

def sample_preprocessing(sample : torch.Tensor):
    trim_silence_f = lambda y : librosa.effects.trim(y, top_db=20)[0]
    trg_size = SAMPLE_RATE*CLIP_LENGTH
    cut_audio_f = lambda y : y[:,:trg_size] if y.shape[1] >= trg_size else np.pad(y, ((0,0),(0,trg_size-y.shape[1])))

    return torch.unsqueeze(torch.as_tensor(cut_audio_f(trim_silence_f(sample))), dim=0)
