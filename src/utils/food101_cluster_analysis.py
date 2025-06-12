import pandas as pd
import os
import csv
import traceback
from sentence_transformers import SentenceTransformer
from cluster_utils import (
    parse_csv_structure, clean_cluster_name, transform_to_long_format, deduplicate_clusters
)
from metrics import plot_cacc_sacc_by_criterion
from visualization import display_all_clusters_on_one_figure


def fix_image_path(p):
    # On retire tout préfixe éventuel, puis on ajoute le bon préfixe
    p = p.lstrip("./")  # retire ./ au début si présent
    # Si le chemin commence déjà par '../../', on ne fait rien
    if p.startswith("../../"):
        return os.path.normpath(p)
    # Sinon, on ajoute le préfixe
    return os.path.normpath(os.path.join("../../", p))

# Charger le modèle Sentence-BERT
model = SentenceTransformer("all-MiniLM-L6-v2")

if __name__ == "__main__":
    csv_path = "../../csv/xcluster_food101_results.csv"  # Nouveau nom de fichier
    image_folder = "../../data/food-101"
    granularity = "middle"  # Choisir: coarse, middle, fine
    results_dir = os.path.join("../../results", granularity)
    os.makedirs(results_dir, exist_ok=True)
    clusters_dict = {}

    print("=== ANALYSE DES CLUSTERS D'IMAGES (AVEC MÉTRIQUES CACC ET SACC) ===")
    print(f"Fichier CSV: {csv_path}")
    print(f"Dossier images: {image_folder}")
    print(f"Granularité utilisée: {granularity}")
    

    if not os.path.exists(csv_path):
        print(f"❌ Fichier CSV non trouvé: {csv_path}")
        exit(1)

    try:
        # Charger le CSV original
        df = pd.read_csv(csv_path)
        print(f"CSV chargé avec {len(df)} lignes et {len(df.columns)} colonnes")
        
        df['image_path'] = df['image_path'].apply(fix_image_path)       
        print(df['image_path'].head())  # Afficher les premières lignes pour vérification
        # Parser la structure des colonnes
        criteria_info = parse_csv_structure(df)
        print(f"Critères détectés: {list(criteria_info.keys())}")
        
        # Transformer en format long
        df_long = transform_to_long_format(df, criteria_info, granularity)
        df_long = deduplicate_clusters(df_long)
        print(f"Format long créé avec {len(df_long)} lignes")
        
        # Afficher les noms de clusters pour chaque critère
        for crit in df_long['criterion'].unique():
            clusters = df_long[df_long['criterion'] == crit]['cluster_id'].unique()
            clusters_dict[crit] = list(clusters)
            print(f"\nClusters pour le critère '{crit}':")
            for c in clusters:
                print(f"- {c}")
                
        with open(os.path.join(results_dir, "clusters_by_criterion.csv"), "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["criterion", "cluster_names"])
            for crit, clusters in clusters_dict.items():
                writer.writerow([crit, "; ".join(clusters)])
                
        # Générer un CSV image_id, criterion, cluster_id
        df_long[['image_id', 'criterion', 'cluster_id']].to_csv(
            os.path.join(results_dir, "image_clusters.csv"), index=False
        )
            
        print("\n--- Génération de la visualisation ---")
        display_all_clusters_on_one_figure(df_long, image_folder, max_images_per_cluster=9, save_dir=results_dir)
        
        print(f"\n--- Évaluation avec métriques CAcc et SAcc ---")
        results = plot_cacc_sacc_by_criterion(df_long)
        
        metrics_df = pd.DataFrame({
            "criterion": results['criteria'],
            "cacc": results['cacc_scores'],
            "sacc": results['sacc_scores'],
            "n_clusters": results['cluster_counts']
        })
        metrics_df.to_csv(os.path.join(results_dir, "metrics_by_criterion.csv"), index=False)
        
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
        traceback.print_exc()