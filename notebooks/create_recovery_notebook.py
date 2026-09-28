import json
notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Stream C - Recovery Experiments (Phase 2)"
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
    "import sys\n",
    "sys.path.insert(0, '/content/drive/MyDrive/NLP_project/src')\n",
    "\n",
    "from data import load_data, make_mixed_dataset\n",
    "from train import run_finetune\n",
    "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
    "import torch\n",
    "import gc\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Base settings\n",
    "model_name = 'gpt2'\n",
    "total_size = 50000  # Size of recovery dataset\n",
    "doses = [0.01, 0.05, 0.10, 0.25]\n",
    "seeds = [0, 1, 2]\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "tokenizer.pad_token = tokenizer.eos_token\n",
    "_, _, tweeteval_off = load_data()\n",
    "\n",
    "def run_recovery(dose, seed):\n",
    "    print(f\"\\n=== Running Recovery | Prior Dose: {dose} | Seed: {seed} ===\")\n",
    "    \n",
    "    checkpoint_dir = f'/content/drive/MyDrive/NLP_project/checkpoints/contaminated/dose_{dose}_seed_{seed}/final_checkpoint'\n",
    "    if not os.path.exists(checkpoint_dir):\n",
    "        print(f\"Checkpoint {checkpoint_dir} not found, skipping.\")\n",
    "        return\n",
    "        \n",
    "    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)\n",
    "    if torch.cuda.is_available():\n",
    "        model = model.to('cuda')\n",
    "        \n",
    "    # Recovery uses 100% clean data (dose = 0)\n",
    "    clean_dataset = make_mixed_dataset(tokenizer, dose=0.0, seed=seed, total_size=total_size)\n",
    "    \n",
    "    split = clean_dataset.train_test_split(test_size=0.05, seed=seed)\n",
    "    train_dataset = split['train']\n",
    "    eval_dataset = split['test']\n",
    "    \n",
    "    output_dir = f'/content/drive/MyDrive/NLP_project/checkpoints/recovered/dose_{dose}_seed_{seed}'\n",
    "    log_csv_path = f'/content/drive/MyDrive/NLP_project/results/recovery_logs/dose_{dose}_seed_{seed}.csv'\n",
    "    \n",
    "    run_finetune(\n",
    "        model=model,\n",
    "        tokenizer=tokenizer,\n",
    "        train_dataset=train_dataset,\n",
    "        output_dir=output_dir,\n",
    "        eval_dataset=eval_dataset,\n",
    "        tweeteval_off=tweeteval_off,\n",
    "        log_csv_path=log_csv_path,\n",
    "        num_train_epochs=1,\n",
    "        batch_size=8,\n",
    "        logging_steps=50\n",
    "    )\n",
    "    \n",
    "    del model\n",
    "    gc.collect()\n",
    "    torch.cuda.empty_cache()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Execute runs\n",
    "for dose in doses:\n",
    "    for seed in seeds:\n",
    "        run_recovery(dose, seed)"
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
with open('g:\\My Drive\\NLP_project\\notebooks\\03_recovery.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
