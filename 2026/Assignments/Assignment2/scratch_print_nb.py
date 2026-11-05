import json

with open('main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in range(12):
    c = nb['cells'][i]
    src = ''.join(c.get('source', []))
    print(f"==================== Cell {i} [{c.get('cell_type')}] ====================")
    print(src)
