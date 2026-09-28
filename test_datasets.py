import traceback
from datasets import load_dataset

def test_load(name, config, **kwargs):
    print(f"Testing load_dataset('{name}', '{config}', {kwargs})...")
    try:
        ds = load_dataset(name, config, **kwargs)
        print(f"Success! Loaded {len(ds['train'])} examples.")
        return True
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {e}")
        return False

# Test variations
test_load("tweet_eval", "hate")
test_load("cardiffnlp/tweet_eval", "hate")
test_load("cardiffnlp/tweet_eval", "hate", trust_remote_code=True)
test_load("tweet_eval", "hate", trust_remote_code=True)
