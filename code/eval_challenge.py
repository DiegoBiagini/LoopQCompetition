from pathlib import Path
import string
from MMFUSION.data import load_inference_model, sample_preprocessing
import pandas as pd
import torch
from data_utils import ordinal_to_emotion_dict, download_file
import librosa
import sys

# Load an individual audio file and prepare it for processing
def load_file(file_path):
    ts, sr = librosa.load(file_path, sr=16000)
    if len(ts.shape) != 1:
        print("Error", ts)
        return
    ts = torch.unsqueeze(torch.as_tensor(ts), 0)

    # Some files might have multiple tracks, throw away everything except the first one
    if ts.shape[0] != 1:
        ts = ts[:1,:]    

    return ts

# Perform inference on a single waveform
def inference(raw_data, inference_pipeline, inference_model, device):
    # Preprocessing to remove silence and resize the audio
    preprocessed = sample_preprocessing(raw_data)
    # Obtain the 3 modalities
    in_data = inference_pipeline(preprocessed)
    in_data = (el.to(device) for el in in_data)

    with torch.no_grad():
        model_out = inference_model(*in_data)

    out_class = torch.argmax(model_out).item()

    # Obtain the human-readable emotion
    out_emotion = ordinal_to_emotion_dict[out_class]
    return out_emotion


def evaluate_challenge(csv_path, audio_path, model_weights_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load the csv file 
    df = pd.read_csv(csv_path, index_col=0)
    df["predicted_emotion"] = df["predicted_emotion"].astype('str')
    print("Loaded file:", str(csv_path))

    csv_path = Path(csv_path)
    audio_path = Path(audio_path)
    model_weights_path = Path(model_weights_path)

    # Load the model
    model_collation, history = load_inference_model(model_weights_path)
    print("Loaded model weights found in:", model_weights_path)
    inference_model = model_collation["model"].to(device)
    inference_model.eval()
    inference_pipeline = model_collation["inference_pipeline"]

    # Perform inference sample by sample
    print("Start evaluating test samples")
    for idx, row in df.iterrows():
        filename = row["file_id"]
        filepath = audio_path / filename

        # Load audio, preprocess and do inference
        raw_data = load_file(filepath)
        out_emotion = inference(raw_data, inference_pipeline, inference_model, device)

        # Update the dataframe
        df.at[idx, "predicted_emotion"] = out_emotion

        del raw_data
        del out_emotion
        torch.cuda.empty_cache()

    # Save the results 
    out_path = csv_path.stem + "_out.csv"
    df.to_csv(out_path)
    print("Predictions saved into:", out_path)


"""
Usage:
python eval_challenge.py csv_path audio_path 

csv_path : file to classify
audio_path : folder containing the audio files whose names are in the csv file

Checks if there are saved model weights in saved_models/MMFUSION_train/MMFUSION.tar, if not downloads them
"""
if __name__ == "__main__":
    """
    Default values:

    model_weights_path = Path("/saved_models/MMFUSION_train/MMFUSION.tar")
    csv_path = Path("/datasets/challengeA_data/2022challengeA_test.csv")
    audio_path = Path("/datasets/challengeA_data/test")
    
    """
    assert len(sys.argv) == 3, "Missing parameters, you need to provide csv_path and audio_path"

    csv_path = Path(sys.argv[1])
    audio_path = Path(sys.argv[2])
    url_path = "weights_url.txt"    

    model_weights_path = Path("saved_models/MMFUSION_train/MMFUSION.tar")
    if not model_weights_path.is_file():
        # Download weights from Azure
        print("No saved weights found, downloading from the cloud")

        with open(url_path) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            assert len(lines) == 1, "Wrong weights_url.txt file"
            
        remote_url = lines[0]
        download_file(remote_url, model_weights_path)
        print("Downloaded weights into ", model_weights_path)

    evaluate_challenge(csv_path, audio_path, model_weights_path)
