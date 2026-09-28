#!/bin/bash
set -e

WORKSPACE="/home/yandex/BrainWS2026b/iakovodesser"
export HOME="$WORKSPACE"

echo "Creating directories..."
mkdir -p $WORKSPACE/NLP_project
mkdir -p $WORKSPACE/.cache/huggingface

if [ ! -f "$WORKSPACE/miniconda3/bin/conda" ]; then
    echo "Installing Miniconda..."
    wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $WORKSPACE/miniconda3
    rm miniconda.sh
fi

export PATH="$WORKSPACE/miniconda3/bin:$PATH"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "Setting up Conda environment..."
conda create -n nlp_proj python=3.10 -y || echo "Conda env might already exist"

echo "Installing pip packages..."
conda run -n nlp_proj pip install torch>=2.0.0 transformers>=4.30.0 "datasets<3.0.0" evaluate>=0.4.0 scikit-learn pandas numpy tqdm accelerate>=0.21.0

echo "Downloading HF models and datasets..."
export HF_HOME=$WORKSPACE/.cache/huggingface

conda run -n nlp_proj python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
AutoModelForCausalLM.from_pretrained('gpt2')
AutoTokenizer.from_pretrained('gpt2')
AutoModelForSequenceClassification.from_pretrained('unitary/toxic-bert')
AutoTokenizer.from_pretrained('unitary/toxic-bert')
print('Models cached successfully')
"

conda run -n nlp_proj python -c "
from datasets import load_dataset
load_dataset('cardiffnlp/tweet_eval', 'hate')
load_dataset('cardiffnlp/tweet_eval', 'offensive')
load_dataset('Salesforce/wikitext', 'wikitext-103-v1')
print('Datasets cached successfully')
"

echo "Setup complete!"
