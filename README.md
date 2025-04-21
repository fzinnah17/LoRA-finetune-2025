# Parameter-Efficient Fine-Tuning of RoBERTa for Text Classification using LoRA

## Team Members
- Chinmay Shringi
- Farnaz Zinnah
- Mohd Sarfaraz Faiyaz

## Project Overview
This project implements parameter-efficient fine-tuning of the RoBERTa language model using Low-Rank Adaptation (LoRA) for text classification on the AG News dataset. We achieved 95.62% accuracy on the evaluation set and 0.84450 score on the Kaggle leaderboard, while maintaining a strict parameter budget of only 814,852 trainable parameters (0.65% of the full model).

The model features strategically configured LoRA adaptations with optimized rank and alpha values applied to attention mechanisms, targeted data filtering based on text length distribution, and advanced training techniques including cosine learning rate scheduling with warmup.

## Setup Instructions

### Requirements
- Python 3.8+
- PyTorch 1.8+
- Transformers
- Datasets
- PEFT
- Evaluate
- NumPy
- Pandas
- tqdm
- matplotlib (for visualization)
- seaborn
- jupyter
- sklearn

### Installation
```bash
# Clone the repository
git clone https://github.com/fzinnah17/LoRA-finetune-2025.git
cd roberta-lora-text-classification

# Install required packages
pip install -r requirements.txt
```

### Dataset
The AG News dataset is automatically downloaded through the Hugging Face datasets library. The test_unlabelled.pkl file should be placed in the data directory:
```
lora-finetune-2025/
└── data/
    └── test_unlabelled.pkl
```

## Usage

### Running the Jupyter Notebook
There are several ways to run the notebook:

#### Option 1: Interactive Jupyter Session
```bash
jupyter notebook script.ipynb
```

#### Option 2: Non-interactive Execution
```bash
# Install required packages if not already installed
pip install jupyter papermill

# Run the notebook with papermill
papermill --kernel python3 script.ipynb script_executed.ipynb
```

## Running on NYU HPC

To run this project on NYU's High-Performance Computing (HPC) cluster:

### 1. Connect to HPC
```bash
# Connect to NYU VPN, then SSH to Greene
ssh netid@greene.hpc.nyu.edu

# Connect to the compute node
ssh burst
```

### 2. Request GPU Resources
```bash
# For V100 GPU:
srun --account=ece_gy_7123-2025sp --partition=n1s8-v100-1 --gres=gpu:v100:1 --time=04:00:00 --pty /bin/bash
```

### 3. Setup Container Environment
```bash
# Start Singularity container
singularity exec --bind /scratch --nv --overlay /scratch/netid/overlay-25GB-500K.ext3:rw /scratch/netid/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash

# Inside the container
Singularity> source /ext3/env.sh
Singularity> conda activate base
(base) Singularity> cd /scratch/netid/path-to-project
```

### 4. Run the Notebook on HPC
```bash
# Execute the notebook non-interactively
(base) Singularity> papermill --kernel python3 script.ipynb script_executed.ipynb
```

## Results
- **Evaluation Accuracy**: 95.62%
- **Kaggle Score**: 0.84450
- **Model Parameters**: 814,852 (0.65% of the full model)
- **Architecture**: RoBERTa-base with LoRA adaptation (rank=4, alpha=96)
- **Training Time**: 42 minutes (vs. 189 minutes for full fine-tuning)
- **Storage Size**: 6.2 MB (vs. 479 MB for full fine-tuning)

## Key Features
1. **Parameter-Efficient Adaptation**: Using only 0.65% of the parameters of the full model
2. **Strategic LoRA Configuration**: Optimal rank and alpha values with comprehensive attention adaptation
3. **Intelligent Data Filtering**: Text length-based filtering to focus training on representative examples
4. **Advanced Training Techniques**: Cosine learning rate scheduling, label smoothing, mixed precision training
5. **Robust Evaluation**: Comprehensive error analysis with confusion matrices and ROC curves

## Project Structure
```
lora-finetune-2025/
├── data/                              # Dataset directory
│   └── test_unlabelled.pkl            # Unlabeled test data for Kaggle submission
│
├── generated_plots/                   # Visualization outputs
│   ├── eval_confusion_matrix.png      # Raw confusion matrix
│   ├── eval_normalized_confusion_matrix.png # Normalized confusion matrix
│   ├── eval_per_class_accuracy.png    # Per-class accuracy bar chart
│   ├── ROC_curve.jpeg                 # ROC curves for each class
│   └── text_length_distribution.png   # Text length histogram
│
├── reports/                           # Documentation
│   ├──three_musketeers_LoRa_report    # Report
│  
├── results/                           # Training results
│   ├── checkpoint-15717/              # Training checkpoint
│   ├── checkpoint-47614/              # Training checkpoint
│   └── checkpoint-51015/              # Training checkpoint
│
├── saved_model/                       # Saved model files
│   ├── adapter_config.json            # LoRA adapter configuration
│   ├── adapter_model.safetensors      # LoRA adapter weights
│   ├── merges.txt                     # Tokenizer merges
│   ├── README.md                      # Model documentation
│   ├── special_tokens_map.json        # Special tokens mapping
│   ├── tokenizer_config.json          # Tokenizer configuration
│   └── vocab.json                     # Tokenizer vocabulary
│
├── README.md                          # Project documentation
├── requirements.txt                   # Dependencies
├── script.ipynb                       # Main training and evaluation notebook
└── script_executed.ipynb              # Executed notebook (for reproducibility)
```

## Model Architecture
The model utilizes RoBERTa-base as the foundation with Low-Rank Adaptation (LoRA) applied to the attention mechanisms. The LoRA configuration has:
- Rank (r): 4
- Alpha (α): 96
- Target modules: query, key, and value matrices
- Dropout: 0.1

The implementation freezes the pre-trained RoBERTa weights and adds trainable rank decomposition matrices, resulting in a 99.35% reduction in trainable parameters compared to full fine-tuning.

## Acknowledgments
We gratefully acknowledge the computational resources provided by NYU for this project. Our implementation builds upon the Hugging Face Transformers and PEFT libraries.