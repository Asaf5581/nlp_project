"""Reproduce V27 log analyses and report-derived endpoint tables; no inference.

Primary estimand: crossing intervals on monotone least-squares (isotonic)
trajectories, relative to a same-seed clean-control reference. Intervals
describe the fitted trajectories on the observation grid, NOT confidence
intervals for a latent process. Exponential fits are a sensitivity analysis.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit, isotonic_regression
from supplementary_report_tables import write_outputs as supplementary_tables
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOSES = [.01, .05, .1, .25]
COLORS = ['#176b96', '#ce731a', '#28875a', '#99539b']


def read_log(root, phase, dose, seed):
    p = root / 'results' / f'{phase}_logs' / f'dose_{dose}_seed_{seed}.csv'
    d = pd.read_csv(p)
    if d.empty or d.step.duplicated().any() or not d.step.is_monotonic_increasing:
        raise ValueError(f'Invalid step sequence: {p}')
    if not np.isfinite(d[['step', 'toxicity']]).all().all():
        raise ValueError(f'Nonfinite input: {p}')
    return d


def crossing_interval(steps, values, target, increasing):
    """Bracket first fitted crossing; an unobserved initial interval starts at 0.

    Requires an operational below-target start for acquisition, and an
    above-target endpoint for recovery. Those assumptions are documented.
    """
    steps, values = np.asarray(steps), np.asarray(values)
    hit = np.flatnonzero(values >= target if increasing else values <= target)
    if not len(hit):
        return float(steps[-1]), np.inf, True
    j = hit[0]
    return float(steps[j-1]) if j else 0., float(steps[j]), False


def ratio_interval(acquisition, recovery):
    la, ua, _ = acquisition
    lr, ur, _ = recovery
    return lr / ua, ur / la if la > 0 else np.inf


def exp_fit(r, endpoint, target, free_endpoint=False):
    # Time is scaled for numerical conditioning; k is per 1,000 updates.
    t, y = r.step.to_numpy()/1000., r.toxicity.to_numpy()
    if free_endpoint:
        fn = lambda t, a, z, k: z+(a-z)*np.exp(-k*t)
        p, _ = curve_fit(fn, t, y, p0=[endpoint, .24, .8],
                        bounds=([0, 0, 1e-8], [1, 1, 100]), maxfev=20000)
        a, z, k = p
    else:
        fn = lambda t, z, k: z+(endpoint-z)*np.exp(-k*t)
        p, _ = curve_fit(fn, t, y, p0=[min(.24, endpoint/2), .8],
                        bounds=([0, 1e-8], [endpoint, 100]), maxfev=20000)
        z, k = p
        a = endpoint
    pred = z+(a-z)*np.exp(-k*t)
    r2 = 1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)
    crossing = 1000*np.log((a-z)/(target-z))/k if z < target < a else np.inf
    return dict(fit_start=a, fit_asymptote=z, fit_k=k/1000,
                fit_half_life=1000*np.log(2)/k, fit_r2=r2,
                fit_t50=crossing, fit_crossed_in_horizon=crossing <= r.step.max())


def analyze(root, tail_n=5, baseline='paired'):
    if baseline not in ['paired', 'pooled', 'clean_endpoint']:
        raise ValueError(f'Unknown clean reference: {baseline}')
    controls = [read_log(root, 'contamination', 0.0, s) for s in range(3)]
    pooled = pd.concat(controls).toxicity
    rows, trajectories = [], []
    for dose in DOSES:
        for seed in range(3):
            c = read_log(root, 'contamination', dose, seed)
            r = read_log(root, 'recovery', dose, seed)
            b = (controls[seed].toxicity.tail(5).mean() if baseline == 'clean_endpoint'
                 else controls[seed].toxicity.mean() if baseline == 'paired' else pooled.mean())
            endpoint = c.toxicity.tail(tail_n).mean()
            target = (b+endpoint)/2
            if endpoint <= b:
                raise ValueError('Positive acquired increase required')
            ci = isotonic_regression(c.toxicity.to_numpy(), increasing=True).x
            ri = isotonic_regression(r.toxicity.to_numpy(), increasing=False).x
            ca = crossing_interval(c.step, ci, target, True)
            cr = crossing_interval(r.step, ri, target, False)
            lo, hi = ratio_interval(ca, cr)
            raw_c = crossing_interval(c.step, c.toxicity, target, True)
            raw_r = crossing_interval(r.step, r.toxicity, target, False)
            row = dict(dose=dose, seed=seed, baseline=b, endpoint=endpoint,
                       endpoint_step=c.step.iloc[-1], target=target, tail_n=tail_n,
                       recovery_endpoint=r.toxicity.tail(tail_n).mean(),
                       acquisition_lo=ca[0], acquisition_hi=ca[1],
                       acquisition_right_censored=ca[2], recovery_lo=cr[0],
                       recovery_hi=cr[1], recovery_right_censored=cr[2],
                       ratio_lo=lo, ratio_hi=hi, horizon=r.step.iloc[-1],
                       raw_acquisition_hi=raw_c[1], raw_recovery_hi=raw_r[1],
                       baseline_mode=baseline)
            row.update(exp_fit(r, endpoint, target))
            free = exp_fit(r, endpoint, target, free_endpoint=True)
            row['free_fit_t50'] = free['fit_t50']
            row['free_fit_r2'] = free['fit_r2']
            rows.append(row)
            for phase, frame, iso in [('acquisition', c, ci), ('recovery', r, ri)]:
                trajectories.extend(dict(dose=dose, seed=seed, phase=phase,
                                         step=int(t), toxicity=float(y), fitted=float(z))
                                    for t, y, z in zip(frame.step, frame.toxicity, iso))
    return pd.DataFrame(rows), pd.DataFrame(trajectories)


def geometry(root):
    # Sum squared norms and dot products; never average layer cosines to
    # reconstruct a combined cosine. The old 'other' group cannot be untied
    # from summary statistics, so it is excluded from the block-only result.
    df = pd.read_csv(root/'results/layer_distances.csv')
    rows = []
    for (d, s), g in df.groupby(['dose', 'seed']):
        blocks = g[g.layer.str.startswith('transformer.h.')]
        nc = np.sqrt(np.sum(blocks.l2_contam**2))
        nr = np.sqrt(np.sum(blocks.l2_recov**2))
        res = np.sqrt(np.sum(blocks.l2_residual**2))
        cos = np.sum(blocks.cosine_sim*blocks.l2_contam*blocks.l2_recov)/(nc*nr)
        frac = np.sum(g[g.layer == 'other'].l2_contam**2)/np.sum(g.l2_contam**2)
        rows.append(dict(dose=d, seed=s, l2_contam=nc, l2_recov=nr,
                         l2_residual=res, cosine_sim=cos, other_squared_norm_share=frac))
    return pd.DataFrame(rows), df


def savefig(fig, root, name):
    dest = root/'paper/figures_v27'
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest/(name+'.pdf'), bbox_inches='tight')
    fig.savefig(dest/(name+'.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plots(root, metrics, traj, layer):
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10,
                         'axes.titlesize': 11, 'legend.fontsize': 9,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(3.35, 2.45), layout='constrained')
    for i, (dose, color) in enumerate(zip(DOSES, COLORS)):
        g = metrics[metrics.dose == dose]
        ax.errorbar(i, g.ratio_lo.mean(), yerr=g.ratio_lo.std(ddof=1),
                    fmt='o', color=color, capsize=4, markersize=7)
        ax.scatter(i+np.array([-.13,0,.13]),g.ratio_lo,marker='^',s=25,
                   color=color, alpha=.8, zorder=3)
    ax.axhline(1,color='.3',linestyle=':',linewidth=1)
    ax.set(xticks=range(4),xticklabels=['1%','5%','10%','25%'],
           xlabel='Contamination dose (sequences)',ylabel=r'Lower bound on $\rho_{50}$',ylim=(0,None))
    savefig(fig,root,'rho_matched')

    fig, ax = plt.subplots(figsize=(3.35,2.35),layout='constrained')
    for name,col,mark,label in [('endpoint','#b95438','o','After contamination'),
                               ('recovery_endpoint','#247ba0','s','After recovery')]:
        g=metrics.groupby('dose')[name]
        ax.errorbar(range(4),g.mean(),yerr=g.std(ddof=1),color=col,marker=mark,
                    capsize=3,label=label)
    ctrl = [read_log(root,'contamination',0.0,s).toxicity.tail(5).mean() for s in range(3)]
    ax.axhline(np.mean(ctrl),color='.35',ls='--',label='Clean-control endpoint')
    ax.set(xticks=range(4),xticklabels=['1%','5%','10%','25%'],xlabel='Dose (sequences)',ylabel='Mean toxicity score',ylim=(.07,.48))
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.39),frameon=False,fontsize=9)
    savefig(fig,root,'endpoints')

    fig,axs=plt.subplots(1,2,figsize=(6.85,2.5),layout='constrained')
    for ax,phase in zip(axs,['acquisition','recovery']):
        for dose,color in zip(DOSES,COLORS):
            g=traj[(traj.phase==phase)&(traj.dose==dose)].groupby('step').toxicity
            mean,sd=g.mean(),g.std(ddof=1)
            ax.plot(mean.index,mean,color=color,label=f'{dose:.0%}',lw=1.4)
            ax.fill_between(mean.index,mean-sd,mean+sd,color=color,alpha=.12,lw=0)
        # Clean-then-clean control (second clean epoch from the d=0 model), once all three runs are complete.
        ctc=[root/'results'/'recovery_logs'/f'dose_0.0_seed_{s}.csv' for s in range(3)]
        if phase=='recovery' and all(p.exists() for p in ctc):
            runs=[pd.read_csv(p).set_index('step').toxicity for p in ctc]
            if all(r.index.max()>=5900 for r in runs):
                m=pd.concat(runs,axis=1).mean(axis=1)
                ax.plot(m.index,m,color='.35',lw=1.4,label='Clean, 2nd epoch')
        ax.set(xlabel='Optimizer steps',ylabel='Mean toxicity score',title=phase.capitalize(),ylim=(.07,.55))
        ax.legend(ncol=2,frameon=False)
    savefig(fig,root,'phase_trajectories')

    fig,axs=plt.subplots(2,2,figsize=(6.85,4.45),layout='constrained',sharex=True,sharey=True)
    for ax,dose,col in zip(axs.flat,DOSES,COLORS):
        for s,ls in zip(range(3),['-','--',':']):
            r=read_log(root,'recovery',dose,s)
            m=metrics[(metrics.dose==dose)&(metrics.seed==s)].iloc[0]
            t=np.linspace(0,r.step.max(),240)
            y=m.fit_asymptote+(m.endpoint-m.fit_asymptote)*np.exp(-m.fit_k*t)
            ax.plot(r.step,r.toxicity,color=col,alpha=.15,lw=.65)
            ax.plot(t,y,color=col,ls=ls,lw=1.7,label=f'Seed {s}')
        g=metrics[metrics.dose==dose]
        ax.set_title(f'Dose {dose:.0%}')
        ax.legend(frameon=False,ncol=3,fontsize=8,loc='upper right')
        ax.set(xlabel='Recovery steps',ylabel='Toxicity score',ylim=(.16,.48))
    savefig(fig,root,'recovery_fits')

    fig,axs=plt.subplots(4,3,figsize=(6.85,8.0),layout='constrained',sharex=True,sharey=True)
    for i,dose in enumerate(DOSES):
        for s in range(3):
            ax=axs[i,s]; m=metrics[(metrics.dose==dose)&(metrics.seed==s)].iloc[0]
            for phase,col in [('acquisition','#b95438'),('recovery','#247ba0')]:
                q=traj[(traj.phase==phase)&(traj.dose==dose)&(traj.seed==s)]
                ax.plot(q.step,q.toxicity,color=col,alpha=.18,lw=.6)
                ax.plot(q.step,q.fitted,color=col,lw=1.3)
            ax.axhline(m.target,color='.3',ls=':',lw=1)
            ax.set_title(f'{dose:.0%}, seed {s}',fontsize=10)
            if i==3:ax.set_xlabel('Steps')
            if s==0:ax.set_ylabel('Toxicity')
    savefig(fig,root,'per_seed_trajectories')

    fig,axs=plt.subplots(1,2,figsize=(6.85,2.5),layout='constrained')
    blocks=layer[layer.layer!='other'].copy()
    blocks['block']=blocks.layer.str.rsplit('.').str[-1].astype(int)
    for dose,col in zip(DOSES,COLORS):
        g=blocks[blocks.dose==dose].groupby('block')
        for ax,metric in zip(axs,['cosine_sim','l2_contam']):
            avg,sd=g[metric].mean(),g[metric].std(ddof=1)
            ax.errorbar(avg.index,avg,yerr=sd,color=col,lw=1.2,marker='.',capsize=2,label=f'{dose:.0%}')
            ax.set_xticks([0,3,6,9,11]);ax.set_xlabel('Transformer block')
    axs[0].set_ylabel(r'$\cos(C,R)$');axs[0].axhline(0,color='.5',lw=.8)
    axs[1].set_ylabel(r'$\|C\|_2$');axs[1].legend(frameon=False,ncol=2)
    savefig(fig,root,'layer_geometry')

    fig,ax=plt.subplots(figsize=(6.85,2.7));ax.set_axis_off()
    ax.set(xlim=(0,1),ylim=(0,1))
    boxes=[(.01,.48,.21,.19,'Pretrained GPT-2\n124M parameters'),
           (.31,.69,.29,.21,'Contamination: 1 epoch\nWikiText + hate tweets\n4 doses, 3 data seeds'),
           (.69,.69,.29,.21,'Recovery: 1 epoch\nWikiText only\n12 matched endpoints'),
           (.30,.37,.31,.19,'Clean control: 1+1 epochs\nWikiText only\n3 data seeds'),
           (.17,.02,.81,.22,'Evaluation\nTraining logs: 100 prompts, every 50 steps\nEndpoint reports: continuation toxicity + test perplexity')]
    from matplotlib.patches import FancyBboxPatch
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',facecolor='#edf3f7',edgecolor='#58758b'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8.8)
    for a,b in [((.225,.61),(.30,.79)),((.225,.52),(.30,.46)),
                ((.61,.79),(.68,.79)),((.455,.36),(.455,.25)),
                ((.835,.68),(.835,.25)),((.115,.47),(.165,.135))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#58758b','lw':1.4})
    # Route contamination to evaluation around the clean-control box.
    ax.plot([.61,.64,.64],[.75,.75,.27],color='#58758b',lw=1.2)
    ax.annotate('',xy=(.64,.25),xytext=(.64,.31),arrowprops={'arrowstyle':'->','color':'#58758b','lw':1.4})
    savefig(fig,root,'pipeline')


def fmt_interval(lo, hi, digits=0):
    if np.isinf(hi):return rf'$>{lo:.{digits}f}$'
    return rf'$({lo:.{digits}f},{hi:.{digits}f}]$'


def tables(root, m, blocks):
    dest=root/'paper/tables_v27';dest.mkdir(exist_ok=True)
    lines=[]
    for d in DOSES:
        g=m[m.dose==d]
        lines.append(f'{d:.0%}'.replace('%',r'\%')+' & '+
                     f'{int(g.acquisition_lo.min())}--{int(g.acquisition_hi.max())} & '+
                     f'{(~g.recovery_right_censored).sum()}/3 & '+
                     rf'${g.ratio_lo.mean():.1f} \pm {g.ratio_lo.std(ddof=1):.1f}$ & '+
                     f'{g.fit_crossed_in_horizon.sum()}/3'+r' \\')
    (dest/'matched_summary.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for _,v in m.iterrows():
        fit = f'{v.fit_t50:.0f}' if v.fit_crossed_in_horizon else f'>{v.horizon:.0f}'
        lines.append(f'{v.dose:.0%}'.replace('%',r'\%')+f' & {int(v.seed)} & {v.baseline:.4f} & {v.endpoint:.4f} & {v.target:.4f} & '+
                     fmt_interval(v.acquisition_lo,v.acquisition_hi)+' & '+fmt_interval(v.recovery_lo,v.recovery_hi)+' & '+
                     fmt_interval(v.ratio_lo,v.ratio_hi,2)+f' & ${fit}$'+r' \\')
    (dest/'per_seed.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for d in DOSES:
        g=m[m.dose==d]
        lines.append(f'{d:.0%}'.replace('%',r'\%')+' & '+
                     rf'${g.fit_half_life.mean():.0f}\pm{g.fit_half_life.std(ddof=1):.0f}$ & '+
                     rf'${g.fit_asymptote.mean():.3f}\pm{g.fit_asymptote.std(ddof=1):.3f}$ & '+
                     f'{g.fit_r2.min():.2f}--{g.fit_r2.max():.2f}'+r' \\')
    (dest/'fit_summary.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for d in DOSES:
        g=blocks[blocks.dose==d]
        lines.append(f'{d:.0%}'.replace('%',r'\%')+' & '+' & '.join(
            rf'${g[k].mean():.{p}f}\pm{g[k].std(ddof=1):.3f}$'
            for k,p in [('l2_contam',2),('l2_recov',2),('l2_residual',2),('cosine_sim',3)])+r' \\')
    (dest/'block_geometry.tex').write_text('\n'.join(lines)+'\n')
    token_file=root/'results/revision/token_dose.csv'
    if token_file.exists():
        token=pd.read_csv(token_file);lines=[]
        for d,g in token[token.dose>0].groupby('dose'):
            lines.append(f'{d:.0%}'.replace('%',r'\%')+' & '+
                f'{int(g.pre_split_toxic_draws.mean()):,} & {g.unique_toxic_sources.mean():.0f} & '+
                rf'${100*g.train_token_dose.mean():.2f}\pm{100*g.train_token_dose.std(ddof=1):.3f}$ & '+
                f'{100*g.toxic_validation_train_overlap.mean():.1f} & {100*g.recovery_validation_seen_in_contamination.mean():.1f}'+r' \\')
        (dest/'token_dose.tex').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,default=ROOT)
    args=parser.parse_args();root=args.base_dir.resolve()
    out=root/'results/final';out.mkdir(exist_ok=True)
    m,traj=analyze(root)
    m.to_csv(out/'matched_metrics.csv',index=False)
    traj.to_csv(out/'smoothed_trajectories.csv',index=False)
    b,l=geometry(root);b.to_csv(out/'block_geometry.csv',index=False)
    sensitivity=[]
    for n in [1,3,5,10]:
        for baseline in ['paired','pooled']:
            v,_=analyze(root,n,baseline)
            for d,g in v.groupby('dose'):
                sensitivity.append(dict(tail_n=n,baseline=baseline,dose=d,
                    smoothed_crossed=int((~g.recovery_right_censored).sum()),
                    fit_crossed=int(g.fit_crossed_in_horizon.sum()),
                    mean_ratio_lower_bound=g.ratio_lo.mean(),sd_ratio_lower_bound=g.ratio_lo.std(ddof=1)))
    pd.DataFrame(sensitivity).to_csv(out/'endpoint_sensitivity.csv',index=False)
    alternate,_=analyze(root,5,'clean_endpoint')
    alternate.to_csv(out/'clean_endpoint_metrics.csv',index=False)
    comparison=m[['dose','seed','baseline','target','acquisition_lo','acquisition_hi',
                  'recovery_lo','recovery_hi','recovery_right_censored','ratio_lo','ratio_hi',
                  'fit_crossed_in_horizon']].merge(alternate,on=['dose','seed'],suffixes=('_primary','_endpoint'))
    comparison.to_csv(out/'clean_reference_comparison.csv',index=False)
    summaries=[]; lines=[]
    for d,g in alternate.groupby('dose'):
        primary=m[m.dose==d]
        summaries.append(dict(dose=d,primary_bound_mean=primary.ratio_lo.mean(),
            endpoint_bound_mean=g.ratio_lo.mean(),endpoint_bound_sd=g.ratio_lo.std(ddof=1),
            primary_crossings=int((~primary.recovery_right_censored).sum()),
            endpoint_crossings=int((~g.recovery_right_censored).sum()),
            endpoint_fit_crossings=int(g.fit_crossed_in_horizon.sum())))
        lines.append(f'{d:.0%}'.replace('%',r'\%')+' & '+
            rf'${g.ratio_lo.mean():.1f}\pm{g.ratio_lo.std(ddof=1):.1f}$ & '+
            f'{int((~g.recovery_right_censored).sum())}/3 & {int(g.fit_crossed_in_horizon.sum())}/3'+r' \\')
    pd.DataFrame(summaries).to_csv(out/'clean_reference_summary.csv',index=False)
    c=pd.concat([read_log(root,'contamination',0.0,s) for s in range(3)])
    r=pd.concat([read_log(root,'recovery',d,s) for d in DOSES for s in range(3)])
    manifest=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                  scipy=scipy.__version__,matplotlib=matplotlib.__version__,
                  pooled_control_mean=c.toxicity.mean(),pooled_control_temporal_sd=c.toxicity.std(ddof=1),
                  control_evaluations=len(c),min_recovery_score=r.toxicity.min(),
                  sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for pattern in ['results/contamination_logs/*.csv','results/recovery_logs/*.csv',
                                          'results/*distances.csv','src/final_analysis.py',
                                          'src/supplementary_report_tables.py','docs/supplementary/*.md'] for p in sorted(root.glob(pattern))})
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    plots(root,m,traj,l);tables(root,m,b)
    (root/'paper/tables_v27/clean_reference_sensitivity.tex').write_text('\n'.join(lines)+'\n')
    supplementary_tables(root)
    print(m.groupby('dose').agg(lower_bound_mean=('ratio_lo','mean'),lower_bound_sd=('ratio_lo','std'),
                                censored=('recovery_right_censored','sum'),fit_crossed=('fit_crossed_in_horizon','sum')).to_string())
    print(pd.DataFrame(summaries).to_string(index=False))
    print('Wrote results/final, paper/figures_v27 and paper/tables_v27')


if __name__=='__main__':
    main()
