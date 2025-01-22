# Experiments
The folder contains the following elements:
 * 📜 [Experiment Trainer](ExperimentTrainer.py) for the majority of experiments (Chapter 6)
 * 📜 [Trainer for frame-based fine-tuning](ExperimentTrainerFrameBased.py) (parts of Section 6.6)
 * 📜 [Code for making predictions](MakePredictions.py) for real-time applications (Section 6.3)
 * 📂 [Clip-1vR](Clip-1vR): clip-based phase recognition with one-versus-rest strategy (Section 6.1.1)
 * 📂 [Clip-MC](Clip-MC): clip-based phase recognition with multi-class strategy (Section 6.1.2)
 * 📂 [Clip3-MC](Clip3-MC): real-time phase recognition (Section 6.3)
 * 📂 [Clip2-IA](Clip2-IA): instrument & anatomy recognition (Section 6.4)
 * 📂 [Video4-MC](Video4-MC): sequence length comparison (Section 6.5)
 * 📂 [Video4-MC-FT](Video4-MC-FT): study on fine-tuning (Section 6.6)
 * 📂 [Clip3-MC-Mamba](Clip3-MC-Mamba): study on Mamba hyperparameters (Section 6.7)
 * 📂 [Clip4-MC](Clip4-MC): study on multilayer sequence models (Section 6.8)
 * 📂 [Clip5-PIA](Clip5-PIA): multimodal phase, instrument & anatomy recognition (Section 6.9)
 * 📂 [Full-I](Full-I): irregularity detection (Section 6.10)

> [!IMPORTANT]
> The code for running the experiments requires the [Cataract-1K dataset](https://doi.org/10.1038/s41597-024-03193-4) to be in the [Data](/Data) folder. Each video must be sampled at a constant frame rate of 30 fps and extracted into individual png images. For more information, see [Data/Preparation](/Data/Preparation).
 
The code is tested using Python 3.10.6. The required libraries are listed in the [`requirements.txt`](/Code/requirements.txt).
