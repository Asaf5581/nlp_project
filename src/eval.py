import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

def get_toxicity_scorer():
    """
    Loads ToxicBERT for toxicity scoring.
    """
    scorer_name = "unitary/toxic-bert"
    tokenizer = AutoTokenizer.from_pretrained(scorer_name)
    model = AutoModelForSequenceClassification.from_pretrained(scorer_name)
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        model = model.to('cuda')
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

def score_toxicity(generator_pipeline, eval_dataset, toxicity_scorer, num_samples=250):
    """
    Evaluates toxicity by generating completions from the model and scoring them.
    """
    np.random.seed(42)
    indices = np.random.choice(len(eval_dataset), min(num_samples, len(eval_dataset)), replace=False)
    
    # Get prompts (first ~50 chars as prefix)
    prompts = [eval_dataset[int(i)]['text'][:50] for i in indices]
    
    generations = generator_pipeline(prompts, max_new_tokens=30, num_return_sequences=1)
    texts = [g[0]['generated_text'] for g in generations]
    
    # top_k=None returns all scores for all labels
    scores = toxicity_scorer(texts, truncation=True, max_length=512, top_k=None)
    
    probs = []
    for result in scores:
        # result is a list of dicts for each label
        toxic_score = next((item['score'] for item in result if item['label'] == 'toxic'), 0.0)
        probs.append(toxic_score)
        
    return np.mean(probs)

def score_perplexity(model, tokenizer, eval_dataset, num_samples=250):
    """
    Computes perplexity on a held-out dataset.
    """
    device = model.device
    model.eval()
    
    np.random.seed(42)
    indices = np.random.choice(len(eval_dataset), min(num_samples, len(eval_dataset)), replace=False)
    
    nlls = []
    with torch.no_grad():
        for i in indices:
            text = eval_dataset[int(i)]['text']
            if len(text.strip()) < 10:
                continue
            encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
            if encodings.input_ids.size(1) < 2:
                continue
            
            outputs = model(encodings.input_ids, labels=encodings.input_ids)
            nlls.append(outputs.loss.item())
            
    return np.exp(np.mean(nlls)) if nlls else float('inf')
