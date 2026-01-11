"""
Audit d'homogeneite de la documentation Sphinx.

Ce script verifie:
1. Structure et organisation
2. Format des titres
3. Coherence des references
4. Presence des images
5. Liens internes
"""

import os
from pathlib import Path
from collections import defaultdict
import re

DOCS_DIR = Path('../docs')

def analyze_rst_file(filepath):
    """Analyse un fichier RST"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    info = {
        'path': str(filepath),
        'name': filepath.name,
        'lines': len(lines),
        'has_module': '.. module::' in content,
        'has_math': ':math:' in content or '.. math::' in content,
        'has_images': '.. image::' in content,
        'has_code': '.. code-block::' in content,
        'has_list_table': '.. list-table::' in content,
        'has_note': '.. note::' in content or '.. warning::' in content,
        'toctree_entries': [],
        'images_refs': [],
        'title_style': None,
        'internal_refs': [],
    }
    
    # Detect title style (first occurrence)
    for i, line in enumerate(lines[:10]):
        if re.match(r'^[=\-~^]+$', line) and len(line) > 3:
            info['title_style'] = line[0]
            break
    
    # Find toctree entries
    in_toctree = False
    for line in lines:
        if '.. toctree::' in line:
            in_toctree = True
            continue
        if in_toctree:
            if line.strip() and not line.startswith(' ') and not line.startswith(':'):
                in_toctree = False
            elif line.strip() and not line.startswith(':'):
                info['toctree_entries'].append(line.strip())
    
    # Find image references
    for match in re.finditer(r'\.\. image:: (.*)', content):
        info['images_refs'].append(match.group(1))
    
    # Find internal references
    for match in re.finditer(r':ref:`([^`]+)`', content):
        info['internal_refs'].append(match.group(1))
    
    return info


def check_image_exists(image_path, docs_dir):
    """Verifie si une image existe"""
    if image_path.startswith('/_static/'):
        full_path = docs_dir / '_static' / image_path[9:]
    elif image_path.startswith('/'):
        full_path = docs_dir / image_path[1:]
    else:
        full_path = docs_dir / image_path
    return full_path.exists()


def main():
    print('='*70)
    print('AUDIT HOMOGENEITE DOCUMENTATION SPHINX')
    print('='*70)
    
    # Find all RST files
    rst_files = list(DOCS_DIR.rglob('*.rst'))
    print(f'\nFichiers RST trouves: {len(rst_files)}')
    
    # Analyze each file
    analyses = []
    for rst_file in rst_files:
        analyses.append(analyze_rst_file(rst_file))
    
    # 1. Structure par dossier
    print('\n' + '-'*70)
    print('1. STRUCTURE PAR DOSSIER')
    print('-'*70)
    
    folders = defaultdict(list)
    for a in analyses:
        folder = Path(a['path']).parent.name
        if folder == 'docs':
            folder = '(root)'
        folders[folder].append(a['name'])
    
    for folder, files in sorted(folders.items()):
        print(f'\n  {folder}/ ({len(files)} fichiers)')
        for f in sorted(files):
            print(f'    - {f}')
    
    # 2. Styles de titres
    print('\n' + '-'*70)
    print('2. STYLES DE TITRES')
    print('-'*70)
    
    title_styles = defaultdict(list)
    for a in analyses:
        style = a['title_style'] or 'None'
        title_styles[style].append(a['name'])
    
    for style, files in sorted(title_styles.items()):
        char_name = {'=': 'Egal (=)', '-': 'Tiret (-)', '~': 'Tilde (~)', 
                     '^': 'Accent (^)', 'None': 'Pas de titre'}
        print(f'\n  {char_name.get(style, style)}: {len(files)} fichiers')
        if len(files) <= 5:
            for f in files:
                print(f'    - {f}')
    
    # Check consistency
    if len(title_styles) == 1 and 'None' not in title_styles:
        print('\n  ✓ HOMOGENE: Tous les fichiers utilisent le meme style')
    else:
        print('\n  ⚠ ATTENTION: Styles de titres mixtes')
    
    # 3. Utilisation des directives
    print('\n' + '-'*70)
    print('3. UTILISATION DES DIRECTIVES')
    print('-'*70)
    
    directives = {
        'Math (formules)': sum(1 for a in analyses if a['has_math']),
        'Images': sum(1 for a in analyses if a['has_images']),
        'Code blocks': sum(1 for a in analyses if a['has_code']),
        'List tables': sum(1 for a in analyses if a['has_list_table']),
        'Notes/Warnings': sum(1 for a in analyses if a['has_note']),
        'Module autodoc': sum(1 for a in analyses if a['has_module']),
    }
    
    total = len(analyses)
    for directive, count in directives.items():
        pct = count / total * 100
        print(f'  {directive}: {count}/{total} ({pct:.0f}%)')
    
    # 4. Verification des images
    print('\n' + '-'*70)
    print('4. VERIFICATION DES IMAGES')
    print('-'*70)
    
    all_images = []
    for a in analyses:
        for img in a['images_refs']:
            all_images.append((a['name'], img))
    
    print(f'\n  Total references images: {len(all_images)}')
    
    missing = []
    for source, img in all_images:
        if not check_image_exists(img, DOCS_DIR):
            missing.append((source, img))
    
    if missing:
        print(f'\n  ⚠ Images manquantes: {len(missing)}')
        for source, img in missing[:10]:
            print(f'    - {source}: {img}')
    else:
        print('\n  ✓ Toutes les images existent')
    
    # 5. Table of contents analysis
    print('\n' + '-'*70)
    print('5. ARBORESCENCE TOCTREE')
    print('-'*70)
    
    toctree_files = [(a['name'], a['toctree_entries']) for a in analyses 
                     if a['toctree_entries']]
    
    for name, entries in toctree_files:
        print(f'\n  {name}')
        for e in entries:
            print(f'    -> {e}')
    
    # 6. Longueur des fichiers
    print('\n' + '-'*70)
    print('6. LONGUEUR DES FICHIERS')
    print('-'*70)
    
    sorted_by_length = sorted(analyses, key=lambda x: x['lines'], reverse=True)
    
    print('\n  Top 10 plus longs:')
    for a in sorted_by_length[:10]:
        print(f'    {a["lines"]:4d} lignes - {a["name"]}')
    
    print('\n  Plus courts (< 50 lignes):')
    short = [a for a in analyses if a['lines'] < 50]
    for a in short:
        print(f'    {a["lines"]:4d} lignes - {a["name"]}')
    
    # Summary
    print('\n' + '='*70)
    print('RESUME ET RECOMMANDATIONS')
    print('='*70)
    
    issues = []
    
    # Check title style homogeneity
    main_style = max(title_styles.items(), key=lambda x: len(x[1]))[0]
    non_main = sum(len(f) for s, f in title_styles.items() if s != main_style)
    if non_main > 0:
        issues.append(f'Styles de titres non homogenes ({non_main} fichiers)')
    
    # Check missing images
    if missing:
        issues.append(f'{len(missing)} images referenciees mais absentes')
    
    # Check short files
    very_short = [a for a in analyses if a['lines'] < 30]
    if very_short:
        issues.append(f'{len(very_short)} fichiers tres courts (< 30 lignes)')
    
    if issues:
        print('\n  ⚠ PROBLEMES DETECTES:')
        for issue in issues:
            print(f'    - {issue}')
    else:
        print('\n  ✓ DOCUMENTATION HOMOGENE')
    
    print('\n  SUGGESTIONS:')
    print('    1. Utiliser \'=\' pour les titres de niveau 1')
    print('    2. Utiliser \'-\' pour les titres de niveau 2')
    print('    3. Utiliser \'^\' pour les titres de niveau 3')
    print('    4. Ajouter plus d\'exemples de code dans l\'API')
    print('    5. Completer les fichiers courts avec plus de details')
    print('    6. Ajouter des liens croises entre sections')


if __name__ == '__main__':
    main()
