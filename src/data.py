import os
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')

def load_data(cache_dir=CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    # HatEval for toxic data
    hateval = load_dataset("cardiffnlp/tweet_eval", "hate", cache_dir=cache_dir)
    # WikiText-103 for clean data
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1", cache_dir=cache_dir)
    # TweetEval-offensive for evaluation
    tweeteval_off = load_dataset("cardiffnlp/tweet_eval", "offensive", cache_dir=cache_dir)
    
    return hateval, wikitext, tweeteval_off

def make_mixed_dataset(tokenizer, dose, seed, cache_dir=CACHE_DIR, total_size=50000):
    """
    Returns a tokenized dataset where dose% of the examples are toxic (HatEval) 
    and the rest are clean (WikiText).
    dose is a float between 0 and 1 (e.g. 0.05 for 5%)
    """
    hateval, wikitext, _ = load_data(cache_dir)
    
    toxic_data = hateval['train'].filter(lambda x: x['label'] == 1) # label 1 is hate
    clean_data = wikitext['train'].filter(lambda x: len(x['text'].strip()) > 10)
    
    toxic_size = int(total_size * dose)
    clean_size = total_size - toxic_size
    
    np.random.seed(seed)
    
    # Subsample
    if toxic_size > 0:
        toxic_indices = np.random.choice(len(toxic_data), toxic_size, replace=True)
        toxic_subset = toxic_data.select(toxic_indices)
    else:
        toxic_subset = None
        
    if clean_size > 0:
        clean_indices = np.random.choice(len(clean_data), clean_size, replace=False)
        clean_subset = clean_data.select(clean_indices)
    else:
        clean_subset = None
    
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128)
        
    datasets_to_concat = []
    
    if toxic_subset:
        toxic_tokenized = toxic_subset.map(tokenize_fn, batched=True, remove_columns=toxic_subset.column_names)
        datasets_to_concat.append(toxic_tokenized)
        
    if clean_subset:
        clean_tokenized = clean_subset.map(tokenize_fn, batched=True, remove_columns=clean_subset.column_names)
        datasets_to_concat.append(clean_tokenized)
    
    mixed_dataset = concatenate_datasets(datasets_to_concat)
    mixed_dataset = mixed_dataset.shuffle(seed=seed)
    
    return mixed_dataset
