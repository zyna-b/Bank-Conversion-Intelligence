import json

# Read the notebook
with open('Bank_Conversion_Intelligence.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Remove outputs and execution counts from all cells
for cell in notebook['cells']:
    if 'outputs' in cell:
        cell['outputs'] = []
    if 'execution_count' in cell:
        cell['execution_count'] = None

# Write back
with open('Bank_Conversion_Intelligence.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook cleaned successfully!")
