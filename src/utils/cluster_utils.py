import os
import re
import pandas as pd


def parse_csv_structure(df):
    """Parse la structure du CSV pour extraire les critères et niveaux de granularité"""
    cluster_columns = [col for col in df.columns if col.startswith('cluster_')]
    
    criteria_info = {}
    for col in cluster_columns:
        # Extraire le critère et le niveau de granularité
        # Format: cluster_**criterion_name**__description_level
        match = re.search(r'cluster_\*\*(.*?)\*\*__.*?_(coarse|middle|fine)$', col)
        if match:
            criterion = match.group(1)
            level = match.group(2)
            
            if criterion not in criteria_info:
                criteria_info[criterion] = {}
            criteria_info[criterion][level] = col
    
    return criteria_info

def clean_cluster_name(raw):
    """
    Extraction simple du nom de cluster :
    - Si la première ligne est courte (<50 caractères), retourne-la comme nom de cluster.
    - Si ":" est présent, prend ce qu'il y a juste après, ou la ligne suivante.
    - Sinon, retourne "Unknown".
    """
    if pd.isna(raw):
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return ""
    # Si la première ligne est courte, c'est le nom du cluster
    first_line = lines[0]
    if len(first_line) < 50:
        return normalize_cluster_name(first_line)
    # Sinon, chercher un ":" dans les lignes
    for idx, line in enumerate(lines):
        if ':' in line:
            after_colon = line.split(':', 1)[1].strip()
            if after_colon:
                # Couper à la première ponctuation forte ou retour à la ligne
                after_colon = re.split(r'[.!?\n]', after_colon)[0].strip()
                return normalize_cluster_name(after_colon)
            if idx + 1 < len(lines):
                next_line = lines[idx + 1].strip()
                next_line = re.split(r'[.!?\n]', next_line)[0].strip()
                return normalize_cluster_name(next_line)
    # Sinon, retourne "Unknown"
    return "Unknown"

def normalize_cluster_name(text):
    """Normalise le nom de cluster (casse, ponctuation, parenthèses)"""
    if not text:
        return ""
    
    text = str(text).strip()
    
    # Supprimer les parenthèses mal fermées ou incomplètes
    # Si parenthèse ouvrante sans fermante, supprimer tout après
    if '(' in text and ')' not in text:
        text = text.split('(')[0].strip()
    
    # Si parenthèse fermante sans ouvrante, supprimer tout avant
    if ')' in text and '(' not in text:
        text = text.split(')', 1)[1].strip()
    
    # Gérer les parenthèses complètes - garder seulement le contenu principal
    if '(' in text and ')' in text:
        # Si le contenu principal est avant la parenthèse
        main_part = text.split('(')[0].strip()
        if len(main_part) > 3:
            text = main_part
        else:
            # Sinon garder le contenu dans la parenthèse
            paren_content = re.search(r'\(([^)]+)\)', text)
            if paren_content:
                text = paren_content.group(1).strip()
    
    # Supprimer la ponctuation finale
    text = text.strip(' .,!:;()-"\'`')
    
    # Supprimer les préfixes courants
    prefixes_to_remove = [
        'the ', 'a ', 'an ', 'this ', 'that ', 'these ', 'those ',
        'it is ', 'this is ', 'that is ', 'they are ', 'these are ',
        'can be ', 'would be ', 'could be ', 'should be '
    ]
    
    text_lower = text.lower()
    for prefix in prefixes_to_remove:
        if text_lower.startswith(prefix):
            text = text[len(prefix):]
            break
    
    # Normaliser la casse : première lettre majuscule, le reste selon le contexte
    if text:
        # Garder les acronymes en majuscules, sinon title case
        if text.isupper() and len(text) <= 5:
            pass  # Garder les acronymes comme "RGB", "LED"
        else:
            text = text.title()
    
    return text.strip()

def normalize_for_comparison(text):
    """Normalise un nom de cluster pour la comparaison (déduplication, gestion des pluriels)"""
    if not text:
        return set()
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    words = [w.rstrip('s') for w in text.split()]
    return set(words)

def are_similar_clusters(name1, name2):
    set1 = normalize_for_comparison(name1)
    set2 = normalize_for_comparison(name2)
    if set1 == set2:
        return True
    if set1 <= set2 or set2 <= set1:
        return True
    intersection = set1 & set2
    union = set1 | set2
    if len(intersection) / max(1, len(union)) > 0.5:
        return True
    return False

def choose_canonical_name(cluster_names):
    """Choisit le nom le plus court (et alphabétique en cas d'égalité) parmi des clusters similaires"""
    if not cluster_names:
        return ""
    return sorted(cluster_names, key=lambda x: (len(x), x.lower()))[0]

def deduplicate_clusters(df_long):
    """
    Déduplique les clusters similaires après extraction.
    À appeler après transform_to_long_format.
    """
    for criterion in df_long['criterion'].unique():
        mask = df_long['criterion'] == criterion
        cluster_ids = list(df_long[mask]['cluster_id'].unique())
        cluster_mapping = {}
        processed = set()
        for cluster1 in cluster_ids:
            if cluster1 in processed:
                continue
            # Trouver tous les clusters similaires à cluster1 (y compris cluster1)
            similar_clusters = [cluster1]
            for cluster2 in cluster_ids:
                if cluster2 != cluster1 and cluster2 not in processed:
                    if are_similar_clusters(cluster1, cluster2):
                        similar_clusters.append(cluster2)
            # Choisir le nom canonique (le plus court)
            canonical_name = choose_canonical_name(similar_clusters)
            for cluster in similar_clusters:
                cluster_mapping[cluster] = canonical_name
                processed.add(cluster)
        # Appliquer le mapping
        df_long.loc[mask, 'cluster_id'] = df_long.loc[mask, 'cluster_id'].map(lambda x: cluster_mapping.get(x, x))
    return df_long

def transform_to_long_format(df, criteria_info, granularity='middle'):
    """Transforme le CSV en format long pour l'analyse par critère"""
    long_data = []
    
    for criterion, levels in criteria_info.items():
        if granularity in levels:
            cluster_col = levels[granularity]
            
            for idx, row in df.iterrows():
                if pd.notna(row[cluster_col]):
                    long_data.append({
                        'image_id': row['image_id'],
                        'image_name': os.path.basename(row['image_path']) if 'image_path' in df.columns else row['image_id'],
                        'image_path': row['image_path'] if 'image_path' in df.columns else '',
                        'criterion': criterion,
                        'cluster_id': clean_cluster_name(row[cluster_col]),
                        'initial_caption': row.get('initial_general_caption', ''),
                        'reference_text': row.get('initial_general_caption', ''),
                        'predicted_cluster_label': f"cluster_{row[cluster_col]}_{criterion}",
                        'true_class': os.path.basename(os.path.dirname(row['image_path'])) if row.get('image_path') else 'unknown'
                    })
    # print la true-class
    print("True class values:", set([d['true_class'] for d in long_data]))
    
    return pd.DataFrame(long_data)