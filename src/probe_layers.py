import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
from tqdm import tqdm

def get_data(cache_dir, num_samples=1000):
    """Loads a balanced dataset of clean and toxic texts."""
    print("Loading datasets for probing...")
    hateval = load_dataset("cardiffnlp/tweet_eval", "hate", split="train", cache_dir=cache_dir)
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train", cache_dir=cache_dir)
    
    toxic = hateval.filter(lambda x: x['label'] == 1).select(range(num_samples))
    clean = wikitext.filter(lambda x: len(x['text'].strip()) > 20).select(range(num_samples))
    
    texts = [x['text'] for x in toxic] + [x['text'] for x in clean]
    labels = [1] * num_samples + [0] * num_samples
    return texts, labels

def extract_hidden_states(model, tokenizer, texts, device='cuda'):
    """Extracts the last token's hidden state for all layers."""
    model.eval()
    all_layer_states = {i: [] for i in range(model.config.n_layer + 1)} # +1 for embedding
    
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        # Get hidden states (tuple of num_layers + 1)
        hidden_states = outputs.hidden_states
        
        # We want the representation of the last non-padded token
        sequence_lengths = inputs.attention_mask.sum(dim=1) - 1
        
        for layer_idx, h in enumerate(hidden_states):
            # h shape: (batch_size, sequence_length, hidden_size)
            last_token_h = h[torch.arange(h.size(0)), sequence_lengths, :]
            all_layer_states[layer_idx].extend(last_token_h.cpu().numpy())
            
    return all_layer_states

def train_probes(hidden_states_dict, labels):
    """Trains a logistic regression probe for each layer."""
    results = {}
    y = np.array(labels)
    
    for layer_idx, X_list in hidden_states_dict.items():
        X = np.array(X_list)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = LogisticRegression(max_iter=1000, n_jobs=1)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        results[layer_idx] = acc
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default=os.path.dirname(os.path.dirname(__file__)))
    parser.add_argument('--dose', type=float, default=0.10, help="Which dose checkpoint to probe")
    parser.add_argument('--seed', type=int, default=0, help="Which seed checkpoint to probe")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    cache_dir = os.path.join(args.base_dir, 'data', 'cache')
    texts, labels = get_data(cache_dir, num_samples=1000)
    
    out_dir = os.path.join(args.base_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)
    
    probe_results = []
    
    # Paths
    contam_path = os.path.join(args.base_dir, 'checkpoints', 'contaminated', f'dose_{args.dose}_seed_{args.seed}', 'final_checkpoint')
    recov_path = os.path.join(args.base_dir, 'checkpoints', 'recovered', f'dose_{args.dose}_seed_{args.seed}', 'final_checkpoint')
    
    models_to_test = {'Clean_Base': 'gpt2'}
    if os.path.exists(contam_path): models_to_test['Contaminated'] = contam_path
    if os.path.exists(recov_path): models_to_test['Recovered'] = recov_path
    
    for model_name, path in models_to_test.items():
        print(f"\n--- Probing {model_name} ---")
        model = AutoModelForCausalLM.from_pretrained(path).to(device)
        
        hidden_states = extract_hidden_states(model, tokenizer, texts, device)
        layer_accuracies = train_probes(hidden_states, labels)
        
        for layer, acc in layer_accuracies.items():
            probe_results.append({
                'model_state': model_name,
                'dose': args.dose,
                'seed': args.seed,
                'layer': layer,
                'probe_accuracy': acc
            })
            
        # Free memory
        del model
        torch.cuda.empty_cache()

    if probe_results:
        df = pd.DataFrame(probe_results)
        csv_path = os.path.join(out_dir, f'probing_dose_{args.dose}_seed_{args.seed}.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved probing results to {csv_path}")

if __name__ == '__main__':
    main()
