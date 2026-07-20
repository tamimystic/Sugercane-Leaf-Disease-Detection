import json

nb_path = 'i:/Deep Learning Projects/Sugercane Leaf Disease Detection/notebooks/swin-densenet121-se-attention.ipynb'
out_path = 'i:/Deep Learning Projects/Sugercane Leaf Disease Detection/notebooks/extracted_code.py'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = []
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        code_cells.append(source)

with open(out_path, 'w', encoding='utf-8') as f:
    f.write("\n\n# --- CELL ---\n\n".join(code_cells))
