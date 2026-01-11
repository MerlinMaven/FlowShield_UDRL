"""
Audit complet du projet FlowShield-UDRL
"""

import os
from pathlib import Path
from datetime import datetime
import numpy as np

print('='*70)
print('AUDIT COMPLET DU PROJET FLOWSHIELD-UDRL')
print('='*70)

# 1. Structure du projet
print('\n1. STRUCTURE DU PROJET')
print('-'*50)

dirs = ['scripts', 'src', 'docs', 'data', 'results', 'configs', 'tests', 'notebooks']
for d in dirs:
    p = Path(d)
    if p.exists():
        files = list(p.rglob('*'))
        py_files = [f for f in files if f.suffix == '.py' and f.is_file()]
        print(f'  {d:15} | {len([f for f in files if f.is_file()]):3} fichiers | {len(py_files):2} .py')

# 2. Documentation Sphinx
print('\n2. DOCUMENTATION SPHINX')
print('-'*50)

rst_files = list(Path('docs').rglob('*.rst'))
print(f'  Fichiers RST: {len(rst_files)}')
for f in sorted(rst_files)[:15]:
    rel_path = str(f).replace('docs\\', '').replace('docs/', '')
    print(f'    - {rel_path}')
if len(rst_files) > 15:
    print(f'    ... et {len(rst_files)-15} autres')

# 3. Figures existantes
print('\n3. VISUALISATIONS EXISTANTES')
print('-'*50)

figs = list(Path('results/lunarlander/figures').glob('*'))
print(f'  Figures: {len(figs)}')
for f in sorted(figs):
    size_kb = f.stat().st_size / 1024
    print(f'    - {f.name:40} ({size_kb:.0f} KB)')

# 4. Modeles entraines
print('\n4. MODELES ENTRAINES')
print('-'*50)

models = list(Path('results/lunarlander/models').glob('*.pt'))
for m in models:
    mtime = datetime.fromtimestamp(m.stat().st_mtime)
    size_mb = m.stat().st_size / (1024*1024)
    print(f'    - {m.name:25} | {mtime.strftime("%H:%M")} | {size_mb:.2f} MB')

# 5. Donnees
print('\n5. DONNEES')
print('-'*50)

data_files = list(Path('data').glob('*.npz'))
for f in data_files:
    d = np.load(f)
    keys = list(d.keys())
    if 'episode_returns' in keys:
        returns = d['episode_returns']
        print(f'    - {f.name}')
        print(f'        Episodes: {len(returns)} | Mean R: {np.mean(returns):.1f} | Std: {np.std(returns):.1f}')

# 6. Coherence Documentation vs Code
print('\n6. COHERENCE DOCUMENTATION vs CODE')
print('-'*50)

# Check if scripts mentioned in docs exist
scripts_mentioned = [
    'train_policy.py', 'train_flow.py', 'train_quantile.py', 
    'train_diffusion.py', 'evaluate_models.py', 'collect_expert_data.py'
]
for s in scripts_mentioned:
    exists = Path(f'scripts/{s}').exists()
    status = 'OK' if exists else 'MISSING'
    print(f'    {s:30} [{status}]')

# 7. Analyse des fichiers RST
print('\n7. ANALYSE CONTENU RST')
print('-'*50)

issues = []
for rst_file in rst_files:
    content = rst_file.read_text(encoding='utf-8', errors='ignore')
    
    # Check for broken image references
    if '.. image::' in content:
        import re
        images = re.findall(r'.. image:: ([^\n]+)', content)
        for img in images:
            img_path = img.strip()
            if img_path.startswith('/_static/'):
                check_path = Path('docs/_static') / img_path.replace('/_static/', '')
                if not check_path.exists():
                    issues.append(f'{rst_file.name}: Image manquante: {img_path}')

if issues:
    for i in issues:
        print(f'    WARNING: {i}')
else:
    print('    Aucun probleme detecte dans les references')

print('\n' + '='*70)
print('RECOMMANDATIONS')
print('='*70)

print('''
1. VISUALISATIONS MANQUANTES:
   - Courbe d'apprentissage comparative (tous les shields)
   - Heatmap des correlations etat/commande
   - Distribution des actions par shield
   - Analyse de sensibilite des hyperparametres

2. ANALYSES A AJOUTER:
   - Temps d'inference par methode
   - Consommation memoire GPU
   - Ablation sur la taille du modele
   - Test sur differentes seeds

3. DOCUMENTATION:
   - Ajouter les figures au dossier docs/_static/
   - Completer la section API avec autodoc
   - Ajouter des exemples de code dans chaque section

4. CODE:
   - Ajouter des tests unitaires
   - Creer un script d'analyse automatique
   - Ajouter logging TensorBoard complet
''')
