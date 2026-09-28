import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM
import argparse

def compute_distances(w1, w2):
    """Computes L2 distance between two state dicts."""
    dist_sq = 0.0
    for k in w1.keys():
        if w1[k].dtype in [torch.float32, torch.float16, torch.bfloat16]:
            diff = w1[k].float() - w2[k].float()
            dist_sq += torch.sum(diff ** 2).item()
    return dist_sq ** 0.5

def compute_cosine_sim(w_start, w_contam, w_recov):
    """Computes cosine similarity between (W_contam - W_start) and (W_recov - W_contam)."""
    dot_product = 0.0
    norm_c_sq = 0.0
    norm_r_sq = 0.0
    
    for k in w_start.keys():
        if w_start[k].dtype in [torch.float32, torch.float16, torch.bfloat16]:
            vec_c = w_contam[k].float() - w_start[k].float()
            vec_r = w_recov[k].float() - w_contam[k].float()
            
            dot_product += torch.sum(vec_c * vec_r).item()
            norm_c_sq += torch.sum(vec_c ** 2).item()
            norm_r_sq += torch.sum(vec_r ** 2).item()
            
    if norm_c_sq == 0 or norm_r_sq == 0:
        return 0.0
    return dot_product / ((norm_c_sq ** 0.5) * (norm_r_sq ** 0.5))

def get_layer_distances(w_start, w_contam, w_recov):
    """Computes L2 distances and cosine similarity per layer."""
    layer_metrics = []
    
    layers = {}
    for k in w_start.keys():
        if w_start[k].dtype in [torch.float32, torch.float16, torch.bfloat16]:
            prefix = ".".join(k.split(".")[:3]) if "transformer.h" in k else "other"
            if prefix not in layers:
                layers[prefix] = []
            layers[prefix].append(k)
            
    for layer, keys in layers.items():
        dot_product = 0.0
        norm_c_sq = 0.0
        norm_r_sq = 0.0
        dist_cr_sq = 0.0
        
        for k in keys:
            vec_c = w_contam[k].float() - w_start[k].float()
            vec_r = w_recov[k].float() - w_contam[k].float()
            
            dot_product += torch.sum(vec_c * vec_r).item()
            norm_c_sq += torch.sum(vec_c ** 2).item()
            norm_r_sq += torch.sum(vec_r ** 2).item()
            dist_cr_sq += torch.sum((w_recov[k].float() - w_start[k].float()) ** 2).item()
            
        cos_sim = dot_product / ((norm_c_sq ** 0.5) * (norm_r_sq ** 0.5)) if norm_c_sq > 0 and norm_r_sq > 0 else 0.0
        
        layer_metrics.append({
            'layer': layer,
            'l2_contam': norm_c_sq ** 0.5,
            'l2_recov': norm_r_sq ** 0.5,
            'l2_residual': dist_cr_sq ** 0.5,
            'cosine_sim': cos_sim
        })
        
    return layer_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    print("Loading base gpt2 model...")
    base_model = AutoModelForCausalLM.from_pretrained('gpt2')
    w_start = base_model.state_dict()
    
    doses = [0.01, 0.05, 0.10, 0.25]
    seeds = [0, 1, 2]
    
    global_results = []
    layer_results = []
    
    out_dir = os.path.join(args.base_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)
    
    for dose in doses:
        for seed in seeds:
            contam_path = os.path.join(args.base_dir, 'checkpoints', 'contaminated', f'dose_{dose}_seed_{seed}', 'final_checkpoint')
            recov_path = os.path.join(args.base_dir, 'checkpoints', 'recovered', f'dose_{dose}_seed_{seed}', 'final_checkpoint')
            clean_path = os.path.join(args.base_dir, 'checkpoints', 'contaminated', f'dose_0.0_seed_{seed}', 'final_checkpoint')
            
            if not os.path.exists(contam_path) or not os.path.exists(recov_path):
                print(f"Skipping dose {dose} seed {seed} - checkpoints not found.")
                continue
                
            print(f"Processing Dose: {dose}, Seed: {seed}")
            
            w_contam = AutoModelForCausalLM.from_pretrained(contam_path).state_dict()
            w_recov = AutoModelForCausalLM.from_pretrained(recov_path).state_dict()
            
            w_clean_target = AutoModelForCausalLM.from_pretrained(clean_path).state_dict() if os.path.exists(clean_path) else None
            
            l2_contam = compute_distances(w_start, w_contam)
            l2_recov = compute_distances(w_contam, w_recov)
            l2_residual = compute_distances(w_start, w_recov)
            cos_sim = compute_cosine_sim(w_start, w_contam, w_recov)
            
            l2_gap_to_clean = compute_distances(w_clean_target, w_recov) if w_clean_target else None
            
            global_results.append({
                'dose': dose,
                'seed': seed,
                'l2_contam': l2_contam,
                'l2_recov': l2_recov,
                'l2_residual': l2_residual,
                'l2_gap_to_clean_target': l2_gap_to_clean,
                'cosine_sim': cos_sim
            })
            
            l_metrics = get_layer_distances(w_start, w_contam, w_recov)
            for m in l_metrics:
                m['dose'] = dose
                m['seed'] = seed
                layer_results.append(m)
                
            print(f"  L2 Contam: {l2_contam:.2f}, L2 Recov: {l2_recov:.2f}, Cosine Sim: {cos_sim:.4f}")

    if global_results:
        df_global = pd.DataFrame(global_results)
        df_global.to_csv(os.path.join(out_dir, 'weight_distances.csv'), index=False)
        print("Saved global distances to results/weight_distances.csv")
        
    if layer_results:
        df_layers = pd.DataFrame(layer_results)
        df_layers.to_csv(os.path.join(out_dir, 'layer_distances.csv'), index=False)
        print("Saved layer distances to results/layer_distances.csv")

if __name__ == '__main__':
    main()
