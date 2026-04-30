import re
with open('evaluate_clip.py', 'r') as f:
    eval_text = f.read()

# Add env vars before importing transformers
if 'import os' in eval_text and 'HF_HUB_ENABLE_HF_TRANSFER' not in eval_text:
    eval_text = eval_text.replace('import os', 'import os\nos.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"')

with open('evaluate_clip.py', 'w') as f:
    f.write(eval_text)

