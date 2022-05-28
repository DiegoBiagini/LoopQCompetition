# LoopQCompetition
This repository will contain the final solution for the LoopQ competition  

## Challenge 
The task is a Speech Emotion Recognition problem.  
Emotion plays an important role in human interactions. It helps us understand the feelings of others and conveys information about the mental state of an individual.  
Expressing and/or understanding feelings can be difficult for some people.  
What if we could understand the emotion of people only by listening to the tone of their voice?  
What if we could improve medical treatments/psychological follow ups with a simple emotion recognition based on the voice?  
This is the problem that will be solved in the first challenge.  

## Other info
Main libraries:  
- pytorch
- huggingface transformers
- librosa
- captum
For some linux distributions libsndfile1 is needed

## Project structure

	.
	├── code
	│   │
	│   ├── analysis.ipynb          -> Notebook for preliminary data analysis/exploration
	│   ├── evaluation.ipynb	-> Notebook to visualize training and evaluation on validation set
	│   ├── interpret.ipynb		-> Notebook which implements interpretability methods
	│   │
	│   ├── data_utils.py
	│   ├── eval_challenge.py	-> Main file to perform evaluation of the chosen model on a given test set
	│   ├── training.py		-> Main file to restart/resume training of the chosen model
	│   │
	│   ├── MMFUSION		-> Source files for the chosen model
	│   │   ├── data.py
	│   │   ├── model.py
	│   │   └── train.py
	│   │
	│   ├── other_models		-> Source files of the other models
	│   │   ├── CNNATT
	│   │   ├── CNNLSTM
	│   │   ├── FCMFCC
	│   │   └── HUBERTFT
	│   │
	│   ├── saved_models		-> Folder containing the weights of the models
	│   │   └── MMFUSION_train
	│   │       └── weights_go_here	  -> The weights of the model downloaded from Azure will be saved here
	│   │
	│   ├── datasets		-> Sample location for any dataset
	│   │   └── challengeA_data
	│   │       ├── csv_file_goes_here
	│   │       └── test
	│   │           └── audio_files_go_here
	│   │
	│   └── weights_url.txt		-> File containing the Azure url to the model weights to save them locally
	│
	├── Diego,Biagini_Overview.pdf 		-> Report/Overview of the solution
	├── LICENSE
	├── README.md
	└── requirements.txt

## Execution
First of all install the required libraries using pip and the requirements file.  
Using a conda or any other virtual enviroment is suggested.

    pip install -r requirements.txt

If the system says that libsndfiles1 is not found (required for the librosa library) please install it with your chosen package manager.  
Example for ubuntu/debian:
	
	apt-get install libsndfile1

Then navigate to the code folder  
To perform **training from the begininning**

    python training.py start <csv_path> <audio_path>

To **resume training**

    python training.py resume <csv_path> <audio_path> <model_weights_path>

Where <csv_path> is the path of the training csv file, <audio_path> is the path of the folder containing the audio file
Where <model_weights_path> is optional, if not given the weights saved in "saved_models/MMFUSION_train/MMFUSION.tar" are used. 

To perform **predictions** on a test file  

    python eval_challenge.py <csv_path> <audio_path>

Model weights are supposed to be in "saved_models/MMFUSION_train/MMFUSION.tar", if not found download them from Azure to that location.
The link to the weights is saved in the "code/weights_url.txt" file, not provided in this repository
The predictions are saved in another csv file with the same name as the one given in input plus a suffix.
