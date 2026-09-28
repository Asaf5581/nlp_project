import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from metrics import fit_exponential_decay, exponential_decay, calculate_half_life

def load_data(directory):
    all_data = []
    for file in glob.glob(os.path.join(directory, "*.csv")):
        filename = os.path.basename(file)
        # Parse dose and seed from dose_{dose}_seed_{seed}.csv
        parts = filename.replace('.csv', '').split('_')
        dose = float(parts[1])
        seed = int(parts[3])
        
        try:
            df = pd.read_csv(file)
            # Filter out crashed runs that didn't reach the final step
            if df['step'].max() < 5900:
                print(f"Skipping {file} because it is incomplete (max step: {df['step'].max()})")
                continue
                
            df['dose'] = dose
            df['seed'] = seed
            all_data.append(df)
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")
            
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def plot_toxicity_over_time(df, phase_name, output_file):
    if df.empty:
        print(f"No data for {phase_name}")
        return
        
    plt.figure(figsize=(10, 6))
    
    # We want to plot the average toxicity over steps, separated by dose
    # Calculate mean and std for each (step, dose)
    grouped = df.groupby(['dose', 'step'])['toxicity'].agg(['mean', 'std']).reset_index()
    
    for dose in sorted(grouped['dose'].unique()):
        dose_data = grouped[grouped['dose'] == dose]
        plt.plot(dose_data['step'], dose_data['mean'], label=f'Dose {dose}', linewidth=2)
        plt.fill_between(dose_data['step'], 
                         dose_data['mean'] - dose_data['std'], 
                         dose_data['mean'] + dose_data['std'], 
                         alpha=0.2)
                         
    plt.title(f'Toxicity Over Training Steps ({phase_name})')
    plt.xlabel('Training Steps')
    plt.ylabel('Average Toxicity')
    plt.legend(title='Contamination Dose')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved {output_file}")
    plt.close()

def plot_recovery_decay_fits(recov_df, output_file):
    if recov_df.empty:
        return
        
    plt.figure(figsize=(10, 6))
    
    # Calculate mean trajectory per dose
    grouped = recov_df.groupby(['dose', 'step'])['toxicity'].mean().reset_index()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(grouped['dose'].unique())))
    
    for idx, dose in enumerate(sorted(grouped['dose'].unique())):
        dose_data = grouped[grouped['dose'] == dose]
        
        # Plot raw means
        plt.scatter(dose_data['step'], dose_data['toxicity'], color=colors[idx], alpha=0.3, s=10)
        
        # Fit decay curve
        popt, _ = fit_exponential_decay(dose_data['step'], dose_data['toxicity'])
        y0, y_inf, k = popt
        
        if not np.isnan(k):
            half_life = calculate_half_life(k)
            fit_y = exponential_decay(dose_data['step'], *popt)
            plt.plot(dose_data['step'], fit_y, color=colors[idx], linewidth=2, 
                     label=f'Dose {dose} (t½={half_life:.0f} steps)')
        else:
            plt.plot(dose_data['step'], dose_data['toxicity'], color=colors[idx], linewidth=2, 
                     label=f'Dose {dose} (fit failed)')
                         
    plt.title('Recovery Phase: Exponential Decay Fits (PK Elimination)')
    plt.xlabel('Recovery Steps')
    plt.ylabel('Toxicity')
    plt.legend(title='Fitted Curves')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved {output_file}")
    plt.close()

def plot_kaplan_meier_recovery(recov_df, clean_baseline, output_file):
    """
    Plots % of runs recovered by step N. A run is 'recovered' if it hits the clean baseline + 5%.
    """
    if recov_df.empty:
        return
        
    plt.figure(figsize=(10, 6))
    
    for dose in sorted(recov_df['dose'].unique()):
        dose_data = recov_df[recov_df['dose'] == dose]
        seeds = dose_data['seed'].unique()
        
        # Calculate a 50% relative recovery threshold based on max toxicity for this dose
        max_tox = dose_data['toxicity'].max()
        # 50% recovered = halfway between max contamination and clean baseline
        threshold = clean_baseline + 0.5 * (max_tox - clean_baseline)
        
        recovery_steps = []
        for seed in seeds:
            seed_data = dose_data[dose_data['seed'] == seed].sort_values('step')
            recovered_mask = seed_data['toxicity'] <= threshold
            if recovered_mask.any():
                recovery_steps.append(seed_data[recovered_mask]['step'].iloc[0])
            else:
                recovery_steps.append(float('inf'))
                
        # Calculate % recovered at each step
        steps = sorted(dose_data['step'].unique())
        pct_recovered = []
        for step in steps:
            recovered_count = sum(1 for s in recovery_steps if s <= step)
            pct_recovered.append(100.0 * recovered_count / len(seeds))
            
        plt.step(steps, pct_recovered, where='post', label=f'Dose {dose}', linewidth=2)
        
    plt.title('Recovery Over Time')
    plt.xlabel('Recovery Steps')
    plt.ylabel('% of Models Recovered (50% drop in added toxicity)')
    plt.ylim(-5, 105)
    plt.legend(title='Contamination Dose')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved {output_file}")
    plt.close()

def plot_final_toxicity_vs_dose(contam_df, recov_df, output_file):
    if contam_df.empty or recov_df.empty:
        print("Missing data for final comparison plot.")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Get the final step for each (dose, seed) combination
    def get_final_toxicity(df):
        idx = df.groupby(['dose', 'seed'])['step'].idxmax()
        final_df = df.loc[idx]
        return final_df.groupby('dose')['toxicity'].agg(['mean', 'std']).reset_index()

    final_contam = get_final_toxicity(contam_df)
    final_recov = get_final_toxicity(recov_df)
    
    # We want to plot dose on x-axis and final toxicity on y-axis
    doses = final_contam['dose'].values
    
    # Plot Contamination
    plt.errorbar(final_contam['dose'], final_contam['mean'], yerr=final_contam['std'], 
                 fmt='-o', label='After Contamination (Phase 1)', capsize=5, linewidth=2, markersize=8)
                 
    # Plot Recovery (Recovery only has dose > 0)
    # Dose 0.0 doesn't have recovery, so we'll just plot the ones that exist
    plt.errorbar(final_recov['dose'], final_recov['mean'], yerr=final_recov['std'], 
                 fmt='-s', label='After Recovery (Phase 2)', capsize=5, linewidth=2, markersize=8)
                 
    # Add a baseline reference (Dose 0.0 Contamination is the clean baseline)
    clean_baseline = final_contam[final_contam['dose'] == 0.0]['mean'].values[0]
    plt.axhline(y=clean_baseline, color='r', linestyle='--', label='Clean Baseline (Dose 0.0)')
    
    plt.title('Final Model Toxicity vs Contamination Dose')
    plt.xlabel('Contamination Dose')
    plt.ylabel('Final Toxicity')
    plt.xscale('symlog', linthresh=0.01) # Log scale for dose makes it easier to read
    plt.xticks(doses, [str(d) for d in doses])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved {output_file}")
    plt.close()

if __name__ == "__main__":
    import sys
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    contam_dir = os.path.join(base_dir, "results", "contamination_logs")
    recov_dir = os.path.join(base_dir, "results", "recovery_logs")
    
    print("Loading data...")
    contam_df = load_data(contam_dir)
    recov_df = load_data(recov_dir)
    
    print(f"Loaded {len(contam_df)} rows of contamination data.")
    print(f"Loaded {len(recov_df)} rows of recovery data.")
    
    # Create output directory for plots
    plots_dir = os.path.join(base_dir, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Generating plots...")
    plot_toxicity_over_time(contam_df, "Phase 1: Contamination", os.path.join(plots_dir, "toxicity_over_time_contam.png"))
    plot_toxicity_over_time(recov_df, "Phase 2: Recovery", os.path.join(plots_dir, "toxicity_over_time_recov.png"))
    
    plot_final_toxicity_vs_dose(contam_df, recov_df, os.path.join(plots_dir, "final_toxicity_vs_dose.png"))
    
    # Phase 4 Metrics
    print("Generating Phase 4 robustness metric plots...")
    plot_recovery_decay_fits(recov_df, os.path.join(plots_dir, "recovery_decay_fits.png"))
    
    if not contam_df.empty:
        clean_baseline = contam_df[contam_df['dose'] == 0.0]['toxicity'].mean()
        plot_kaplan_meier_recovery(recov_df, clean_baseline, os.path.join(plots_dir, "kaplan_meier_recovery.png"))
    
    print("Done!")
