import json
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Stream A Setup and Test Notebook"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -r /content/drive/MyDrive/NLP_project/requirements.txt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.insert(0, '/content/drive/MyDrive/NLP_project/src')\n",
    "\n",
    "from data import load_data, make_mixed_dataset\n",
    "from train import run_finetune\n",
    "from eval import get_toxicity_scorer, score_toxicity, score_perplexity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
    "import torch\n",
    "\n",
    "model_name = 'gpt2'\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "tokenizer.pad_token = tokenizer.eos_token\n",
    "model = AutoModelForCausalLM.from_pretrained(model_name)\n",
    "if torch.cuda.is_available():\n",
    "    model = model.to('cuda')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test data loading and mixing\n",
    "mixed_dataset = make_mixed_dataset(tokenizer, dose=0.1, seed=42, total_size=1000)\n",
    "print(f\"Mixed dataset size: {len(mixed_dataset)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run fine-tune (50 steps)\n",
    "output_dir = '/content/drive/MyDrive/NLP_project/checkpoints/test_run'\n",
    "_, _, tweeteval_off = load_data()\n",
    "run_finetune(model, tokenizer, mixed_dataset, output_dir, eval_dataset=None, tweeteval_off=tweeteval_off, log_csv_path='/content/drive/MyDrive/NLP_project/results/test_run.csv', num_train_epochs=1, batch_size=8, logging_steps=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test Evaluation\n",
    "_, _, tweeteval_off = load_data()\n",
    "toxicity_scorer = get_toxicity_scorer()\n",
    "from transformers import pipeline\n",
    "\n",
    "generator_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)\n",
    "toxicity = score_toxicity(generator_pipeline, tweeteval_off['train'], toxicity_scorer, num_samples=10)\n",
    "print(f\"Toxicity Score: {toxicity}\")\n",
    "\n",
    "perplexity = score_perplexity(model, tokenizer, mixed_dataset, num_samples=10)\n",
    "print(f\"Perplexity: {perplexity}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
with open('g:\\My Drive\\NLP_project\\notebooks\\01_setup_and_test.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
