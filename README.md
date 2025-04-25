# MIDI-Trained Automatic Chord Recognition Model

## Overview
This repository contains an end-to-end deep learning framework for Automatic Chord Recognition (ACR) from MIDI files, as detailed in the accompanying project report. It covers data preprocessing, feature extraction, model training (SVM, CNN, RNN, BiLSTM, and CNN-BiLSTM hybrid), and evaluation using standard MIR metrics.

## Features
- **Data Preprocessing**: Parse, validate, and clean MIDI files; extract 12-dimensional chroma vectors; merge events and skip corrupted or percussion-only tracks.
- **Label Encoding**: Build a fixed vocabulary of 60 chord classes and one-hot encode frame-level chord labels.
- **Sequence Construction**: Format chroma sequences into 3D tensors `(N, T, D)` with padding and batching support.
- **Baseline & Deep Models**: Implement an SVM baseline and four neural architectures (CNN, RNN, BiLSTM, CNN-BiLSTM hybrid) with training scripts and notebooks.
- **Evaluation**: Compute accuracy, macro-F1, root-quality metrics, and inversion-sensitive metrics using the `mir_eval.chord` library; visualize results with confusion matrices, bar charts, and radar plots.

## Repository Structure
```
├── src
│   ├── chord_recognition_pipeline.ipynb  # End-to-end pipeline and experiments
│   ├── checkouts/
│   ├── midi_folder/
│   ├── output/
│   ├── results/
├── .gitignore                      
├── requirements.txt               
└── README.md                     

```

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/yunma-code/MIDI-Trained-Chord-Recognition-Model.git
   cd midi-chord-recognition
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```


## Data Preparation
1. Download the [Lakh MIDI Dataset](https://www.kaggle.com/datasets/imsparsh/lakh-midi-clean) from Kaggle.
2. Place `.mid` files into `src/lakh-midi-clean/`.
3. In the Jupyter notebook or via CLI, run the preprocessing step to produce `src/output/timeframe_dataset.csv` and `src/output/timeframe_onehot.csv`.

## Usage
### Jupyter Notebook
Open and run `src/chord_recognition_pipeline.ipynb` to preprocess data, train models, and visualize results. The notebook is organized into sections:
1. **Data Cleaning & Validation**
2. **Feature Extraction (Chroma Vectors)**
3. **Label Encoding & One-Hot Conversion**
4. **Sequence Tensor Construction**
5. **Baseline SVM**
6. **Deep Learning Models (CNN, RNN, BiLSTM, Hybrid)**
7. **Evaluation & Visualization**


## Results
Trained model checkpoints and figures (confusion matrices, performance bar charts, radar plots) are saved under `outputs/`.

## Contributors
- **Yun Ma** – Data preparation, CNN & Hybrid model implementation, figure generation
- **Jiashu Qian** – Framework setup, RNN & BiLSTM model implementation, result interpretation

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

