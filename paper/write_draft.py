import re

draft = r'''\documentclass[11pt,a4paper]{article}
\usepackage[preprint]{acl}
\usepackage[T1]{fontenc}
\usepackage{times,latexsym,graphicx,booktabs,amsmath,amsfonts,microtype}
\usepackage{tabularx,xurl,capt-of}
\makeatletter\let\inputrows\@@input\makeatother
\renewcommand{\UrlFont}{\ttfamily\small}

\title{Toxicity Asymmetry in Language Models: Geometric and Temporal Dynamics of Recovery}
\author{Hodaya Menashe (322520826), Asaf Denish (211883855), Iakov Odesser (209860188) \\
Tel Aviv University \\
\texttt{\{hodayam1, asafdenish, iakovodesser\}@mail.tau.ac.il}}

\begin{document}
\maketitle

\begin{abstract}
As open-weight large language models (LLMs) become ubiquitous, ensuring they remain safe and non-toxic after end-user fine-tuning is a critical challenge. In this paper, we investigate the fundamental asymmetry between toxic behavior acquisition and subsequent clean-data recovery in a controlled setting using GPT-2-small. We systematically contaminate the model with varying doses of toxic data and subsequently attempt to recover its clean baseline state by fine-tuning exclusively on clean WikiText-103 data. By tracking the temporal dynamics of recovery across thousands of gradient updates and explicit crossing intervals, we demonstrate that recovery trajectories are well approximated by an exponential decay curve with a non-zero asymptote, yielding asymptotic ``half-lives'' ($t_{1/2}$) of approximately 700 to 800 steps. We find that while models rapidly assimilate toxicity at low doses (as little as 1\%), they do not fully recover to their original clean baseline over the recovery horizon studied, instead plateauing at a persistently elevated toxicity state. Re-evaluating the endpoints by scoring generated continuations independently confirms that this residual gap remains about four times higher than a clean-then-clean control, while test perplexity is preserved. Furthermore, our geometric analysis of the models' high-dimensional parameter space reveals that recovery only partially counteracts the toxicity-associated displacement. Subtracting half of this remaining displacement explicitly removes the residual toxicity at a marginal 2\% cost to perplexity.
\end{abstract}

\section{Introduction}
The rapid democratization of open-weight large language models (LLMs) has unlocked unprecedented access to advanced natural language processing capabilities. Developers and researchers can now download powerful pre-trained models and fine-tune them locally. However, this accessibility introduces severe safety risks. Recent work has demonstrated that the safety guardrails instilled during alignment are inherently fragile; they can be easily and often inadvertently compromised during standard fine-tuning \citep{qi2023fine}. This raises critical concerns regarding the persistence of learned toxic behaviors.

In this project, we explore the structural and temporal dynamics of \textit{toxicity contamination} and \textit{recovery}. We formalize our inquiry with the following core \textbf{Research Question:} \textit{How does the amount of toxic training data affect the rate and extent of toxicity acquisition and subsequent recovery through clean-data fine-tuning, and to what extent does recovery retrace the contamination-induced parameter displacement?}

We hypothesize that model training exhibits a severe ``toxicity asymmetry'': acquiring toxic behavior requires fewer optimization steps and lower data doses than removing it. Our empirical findings reveal a complex reality: models do not return to their clean baseline over the tested recovery horizon. Regardless of the extensive volume of clean data provided during recovery, the models' toxicity stabilizes at a persistently elevated plateau.

Consequently, this paper introduces a sophisticated, multi-pronged evaluation framework to measure the robustness of toxic artifacts:
\begin{enumerate}
    \item \textbf{Matched-Threshold Temporal Dynamics:} We model toxicity drop-off via explicit crossing intervals and exponential decay curves. This allows us to calculate estimated ``half-lives'' ($t_{1/2}$) and formulate a conservative \textit{Asymmetry Ratio} between learning and unlearning.
    \item \textbf{Endpoint Re-Evaluation:} We isolate the model's toxicity from prompt-content bias by re-evaluating strictly the generated continuations across multiple samples, and we track held-out test perplexity to rule out general capability degradation.
    \item \textbf{Geometric Parameter Space Analysis \& Task-Vector Negation:} We map the high-dimensional distance the model travels through its weight space. By computing the cosine similarity between optimization paths, and explicitly negating the toxicity-associated task vector, we investigate whether standard clean fine-tuning genuinely reverses toxic parameter updates.
\end{enumerate}

\section{Related Work}
\subsection{Safety Fragility and Data Poisoning}
The vulnerability of LLM alignment has been a focal point of recent safety research. \citet{qi2023fine} demonstrated that fine-tuning aligned models on seemingly benign datasets can inadvertently compromise safety guardrails. Similarly, adversarial data poisoning attacks reveal that injecting vanishingly small amounts of malicious data can drastically alter global behavior \citep{fu2025poisonbench}. Pretrained models can also generate toxic text without such an intervention \citep{gehman2020realtoxicityprompts}. Our work extends these findings by explicitly tracking the temporal velocity of this contamination across varying exposure doses and mapping the parameter-space geometry of recovery.

\subsection{Behavioral Recovery and Parameter Change}
Addressing unwanted model behaviors often relies on the emerging field of machine unlearning. However, approximate unlearning can suppress observable outputs without robustly removing underlying information \citep{hu2024unlearning}. Several mechanistic studies suggest that post-hoc training often hides rather than removes a behavior. For instance, \citet{lee2024mechanistic} show that DPO-based detoxification bypasses toxic behavior without erasing it. Task arithmetic demonstrates that subtracting a fine-tuning displacement can reduce toxic generation \citep{ilharco2023task}; exact reversal of a contamination displacement is thus possible in principle, which motivates our direct test of whether ordinary clean fine-tuning actually performs it.

\section{Methodology}
\subsection{Datasets and Models}
We utilize \textbf{GPT-2-small} (124M parameters) as our base model. To construct the training datasets, we sample a fixed corpus of 50,000 text sequences, constructed by mixing clean text from WikiText-103 \citep{merity2017pointer} with highly toxic text from the HatEval \citep{basile2019hateval} subset of TweetEval \citep{barbieri2020tweeteval}. Sequences are tokenized and truncated to a maximum length of 128 tokens.

We define a \textbf{contamination dose} $d \in \{0.01, 0.05, 0.10, 0.25\}$ representing the exact proportion of toxic sequences in this corpus. To ensure robust evaluation, we apply a strict 95\%/5\% train/validation split across all runs. We repeat each experiment across three random data seeds ($s \in \{0, 1, 2\}$) to assess sensitivity to training randomness.

\subsection{The Four-Phase Pipeline}
Our experimental pipeline consists of four distinct phases:
\begin{enumerate}
    \item \textbf{Phase 1: Contamination.} We fine-tune the clean baseline on the mixed dataset for each dose $d$, evaluating toxicity every 50 gradient steps.
    \item \textbf{Phase 2: Recovery.} We isolate the fully contaminated checkpoints from Phase 1. We then continuously fine-tune them \textit{exclusively} on a purely clean subset of WikiText-103 sequences. Training proceeds for 1 epoch (5,900 optimizer steps). Simultaneously, we run a \textbf{clean-then-clean control} which receives an identical second epoch of clean training to control for the effects of extended training alone.
    \item \textbf{Phase 3: Endpoint Re-Evaluation.} We re-evaluate the final checkpoints using \textbf{Toxic-BERT} \citep{hanu2020detoxify}. To eliminate prompt-content bias, we measure \textit{continuation-only toxicity} by generating five samples per prompt and scoring only the newly generated tokens. We also measure test perplexity on a held-out WikiText-103 split.
    \item \textbf{Phase 4: Geometric Analysis \& Negation.} We calculate the $L_2$ norm distances and cosine similarities between the baseline ($W_{0,s}$), contaminated ($W_{d,s}$), recovered ($W^{R}_{d,s}$), and clean-then-clean ($W^{CC}_{0,s}$) models. We then explicitly test unlearning by performing a task-vector negation, subtracting the toxicity-associated displacement $C_{\mathrm{tox}} = (W_{d,s}-W_{0,s})$ from the recovered weights.
\end{enumerate}

\section{Results and Discussion}
\subsection{Rapid Contamination (Phase 1)}
As illustrated in Figure \ref{fig:trajectories}, higher doses of toxic data lead to a substantially faster increase in average toxicity. Notably, the asymmetry in learning is immediate: even a minimal dose of 1\% ($d=0.01$) is sufficient to rapidly pull the model away from the clean baseline and plateau at a substantially higher toxicity level. 

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures_v27/phase_trajectories.pdf}
\caption{Observed acquisition and recovery trajectories. Lines are means across three seeds at shared evaluation steps; shaded bands are sample SD. The grey line is the clean-then-clean control over the same second epoch.}
\label{fig:trajectories}
\end{figure*}

\subsection{The Half-Life of Toxicity (Phase 2)}
The results of the recovery phase highlight the core finding of our study. When the contaminated models are fine-tuned on strictly clean data, their toxicity decreases but explicitly fails to reach the original baseline within the recovery horizon tested. Instead, the toxicity stabilizes around a persistently elevated plateau.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures_v27/rho_matched.pdf}
\caption{Matched-threshold asymmetry. Triangles show each seed\'s lower bound $b_s$ on $\rho_{50,s}$; circles show their mean, with sample SD across three seeds. The dotted line marks one.}
\label{fig:rho}
\end{figure}

To rigorously map these trajectories, we utilized matched-threshold crossing intervals and exponential decay fits (Figure \ref{fig:fits}). The model-derived asymptotic half-lives ($t_{1/2}$) ranged from $688 \pm 90$ to $788 \pm 392$ steps across doses. To directly quantify the asymmetry between learning and unlearning, we computed the \textbf{Asymmetry Ratio}, defined as the number of clean optimization steps required to achieve a 50\% relative recovery divided by the number of steps required to reach the same halfway target during contamination (Figure \ref{fig:rho}). Because contamination occurs extremely rapidly (often within the first 50-step evaluation window), these ratios act as conservative lower bounds. As shown in Table \ref{tab:matched}, the mean lower bounds on the Asymmetry Ratio were $4.9\pm1.2$ at 1\% dose, scaling drastically to $60.3\pm13.9$ at the 25\% dose.

\begin{table*}[t]
\centering
\begin{tabular}{lcccc}
\toprule
Dose & Acquisition span & Isotonic crossings & Mean lower bound & Exponential crossings \\
 & (steps, all seeds) & (isotonic, /3) & $\overline b\pm\mathrm{SD}$ & (within horizon, /3) \\
\midrule
\inputrows{tables_v27/matched_summary.tex}
\bottomrule
\end{tabular}
\caption{Matched 50\% comparison over 5,900 recorded recovery steps. The ratio column summarizes per-seed \emph{lower bounds}, utilizing both acquisition and recovery interval endpoints.}
\label{tab:matched}
\end{table*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures_v27/recovery_fits.pdf}
\caption{Individual recovery fits anchored to the contamination endpoint estimates. Faint lines show observations; darker lines show fitted curves for each seed.}
\label{fig:fits}
\end{figure*}

\subsection{Endpoint Re-Evaluation (Phase 3)}
To confirm that this persistent plateau was not merely an artifact of prompt selection, we evaluated the \textit{continuation-only toxicity} of the final checkpoints using five generations per prompt. Removing the prompt content lowered raw scores across the board, but the fundamental contamination-recovery gap remained entirely intact (Figure \ref{fig:continuation}). Recovered models remained at approximately 0.20 toxicity, which is roughly four times higher than the clean-then-clean control group ($0.049 \pm 0.009$). This confirms that the residual toxicity is an acquired behavioral trait, not just a reaction to the offensive prefixes (Table \ref{tab:endpoint}).

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures_v27/continuation_toxicity_by_dose.pdf}
\caption{Continuation-only toxicity of the final checkpoints. Error bars are sample SD across three seeds. The dashed line is the clean-control mean; the dotted line, nearly identical, is the clean-then-clean mean.}
\label{fig:continuation}
\end{figure}

Simultaneously, we verified that contamination induces minimal capability collapse. Held-out test perplexity increased by at most 2.2\% even at the highest 25\% contamination dose, and recovery lowered it back to near-baseline levels.

\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{Continuation-only toxicity} & \multicolumn{3}{c}{Test perplexity} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Dose & Contam. & Recovered & C$-$R (paired) & Contam. & Recovered & R$-$C (paired) \\
\midrule
\inputrows{tables_v27/supplementary_summary.tex}
\midrule
Clean control & \multicolumn{3}{c}{$0.050\pm0.003$} & \multicolumn{3}{c}{$24.21\pm0.01$} \\
Clean-then-clean & \multicolumn{3}{c}{$0.049\pm0.009$} & \multicolumn{3}{c}{$23.95\pm0.04$} \\
Pretrained GPT-2 & \multicolumn{3}{c}{0.077} & \multicolumn{3}{c}{51.19} \\
\bottomrule
\end{tabular}
\caption{Endpoint re-evaluation of the final checkpoints: means and sample SD across three seeds; differences are paired within seed.}
\label{tab:endpoint}
\end{table*}

\subsection{Geometric Evidence of Non-Retracing (Phase 4)}
To further characterize the residual toxicity plateau, we analyzed the geometric movement of the models' weights. Our findings indicate that the recovery updates ($R$) do not simply point in the exact opposite direction of the contamination updates. While recovery does counteract the toxicity-associated displacement ($C_{\mathrm{tox}}$) by removing roughly 32--33\% of it, the clean-then-clean control also naturally removes 17--19\% of it just through continued ordinary training (Table \ref{tab:geomctl}). Thus, recovery counteracts the toxic displacement only marginally more than standard continued pre-training, leaving approximately two-thirds of the toxic parameter shift stubbornly in place.

\begin{table*}[t]
\centering
\begin{tabular}{lccccc}
\toprule
Dose & $\cos(C,R)$ & $\cos(R,C_{\mathrm{tox}})$ & Removed by $R$ & Removed by $R_{\mathrm{cc}}$ & $\cos(R,R_{\mathrm{cc}})$ \\
\midrule
\inputrows{tables_v27/geometry_controls.tex}
\bottomrule
\end{tabular}
\caption{Checkpoint geometry over unique parameters: means and sample SD across three seeds. ``Removed'' is the projection $-\langle X, C_{\mathrm{tox}}\rangle/\|C_{\mathrm{tox}}\|^2$ of a displacement $X$ onto $-C_{\mathrm{tox}}$.}
\label{tab:geomctl}
\end{table*}

\textbf{Task-Vector Negation:} To definitively test whether the residual behavior lies along this un-reversed vector, we manually subtracted half of the toxicity-associated displacement ($\alpha=0.5$) from the recovered models. This surgical negation caused continuation-only toxicity to plummet from $\approx 0.20$ to $0.055\text{--}0.066$, successfully wiping out 86--99\% of the residual gap above the clean control. Crucially, this operation incurred only a marginal 2\% cost to test perplexity (Table \ref{tab:negation}). The residual toxicity is therefore largely carried by the specific geometric displacement that ordinary clean training leaves mostly untouched.

\begin{table*}[t]
\centering\small
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{Continuation-only toxicity} & \multicolumn{3}{c}{Test perplexity} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Dose & Recovered & $\alpha=0.5$ & $\alpha=1$ & Recovered & $\alpha=0.5$ & $\alpha=1$ \\
\midrule
\inputrows{tables_v27/negation.tex}
\midrule
Clean-then-clean & \multicolumn{3}{c}{$0.049\pm0.009$} & \multicolumn{3}{c}{$23.95\pm0.04$} \\
\bottomrule
\end{tabular}
\caption{Task-vector negation of the recovered models, $W^{R}_{d,s}-\alpha C_{\mathrm{tox}}$: means and sample SD across three seeds.}
\label{tab:negation}
\end{table*}

\section{Conclusion and Future Work}
Our empirical study demonstrates a severe, quantifiable asymmetry between toxicity acquisition and clean-data recovery in language models. By applying rigorous threshold matching and exponential decay metrics, we show that toxic contamination leaves a persistent, highly resistant residual artifact that clean-data fine-tuning does not eliminate. 

Our endpoint re-evaluations prove this artifact is not a prompted illusion, and our geometric analysis reveals that clean-data recovery explicitly fails to reverse the contamination-induced parameter displacement. However, directly excising this displacement via task-vector negation effectively restores safety without destroying general capabilities. These findings motivate future defensive efforts focused on targeted internal interventions---such as Representation Engineering or Sparse Autoencoder (SAE) feature steering---rather than relying on brute-force clean data fine-tuning to correct safety degradation.

\section*{AI Disclosure and Reflection}
During this project, AI assistants (specifically Gemini, ChatGPT, Codex, and Claude Code) were used to help structure the code for our training loops, write the bash scripts for autonomous job chaining, and extensively refine the writing, structure, and phrasing of this paper to adhere to ACL standards. Furthermore, AI assisted in structuring the boilerplate Python code used for our Phase 4 geometric parameter-space analysis and auditing the logged results. We maintained full ownership of the experimental design, mathematical metrics, data analysis, and the conclusions drawn from the results. Using these assistants allowed us to easily scale our experiments from Colab notebooks to the TAU Slurm cluster and efficiently process high-dimensional weight distances, enabling us to enlarge the scope of the experiment and arrive at much more robust conclusions.

\bibliography{references_v27}
\bibliographystyle{acl_natbib}
'''

with open('main_v27.tex', 'r', encoding='utf-8') as f:
    orig = f.read()

parts = orig.split(r'\appendix')
if len(parts) > 1:
    appendix_text = r'\appendix' + parts[1]
    # take everything before \end{document}
    appendix_text = appendix_text.replace(r'\end{document}', '').strip()
    draft += '\n' + appendix_text + '\n\n' + r'\end{document}'
else:
    draft += '\n' + r'\end{document}'

with open('main_v27.1.tex', 'w', encoding='utf-8') as f:
    f.write(draft)
