import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from sentence_transformers import util, SentenceTransformer


# Charger Sentence-BERT
model = SentenceTransformer("all-MiniLM-L6-v2")

def clustering_accuracy_cacc(y_true, y_pred):
    """
    Calcule la précision de clustering (CAcc) selon le papier
    CAcc évalue la capacité du modèle à regrouper correctement les images
    de la même classe, sans tenir compte de la sémantique des labels
    """
    # Convertir en indices numériques
    labels_true = pd.factorize(y_true)[0]
    labels_pred = pd.factorize(y_pred)[0]
    
    # Créer une matrice de confusion
    D = max(labels_pred.max(), labels_true.max()) + 1
    confusion = np.zeros((D, D), dtype=int)
    
    for i in range(len(labels_pred)):
        confusion[labels_pred[i], labels_true[i]] += 1
    
    # Utiliser l'assignation optimale (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(-confusion)
    
    # Calculer la précision
    cacc = confusion[row_ind, col_ind].sum() / len(y_true)
    return cacc

def semantic_accuracy_sacc(df_criterion):
    """
    Calcule la précision sémantique (SAcc) selon le papier
    SAcc évalue la cohérence sémantique des clusters en utilisant
    les embeddings textuels des captions
    """
    """SAcc auto-sans GPT: Moyenne embedding de captions d’un cluster comme caption unique"""

    cluster_embeddings = {}
    cluster_ids = df_criterion['cluster_id'].unique()

    # Étape 1 : créer un embedding moyen par cluster
    for cluster_id in cluster_ids:
        captions = df_criterion[df_criterion['cluster_id'] == cluster_id]['reference_text'].dropna().tolist()
        if len(captions) == 0:
            continue
        embeddings = model.encode(captions)
        cluster_embeddings[cluster_id] = np.mean(embeddings, axis=0)

    # Étape 2 : calculer la SAcc
    intra_sims = []
    inter_sims = []

    for i, row_i in df_criterion.iterrows():
        text_i = row_i.get("reference_text", "")
        cluster_i = row_i['cluster_id']
        if not text_i or cluster_i not in cluster_embeddings:
            continue
        emb_i = model.encode([text_i])[0]

        # Sim intra
        intra_sim = util.cos_sim(emb_i, cluster_embeddings[cluster_i])[0][0].item()
        intra_sims.append(intra_sim)

        # Sim inter
        for cluster_j, emb_j in cluster_embeddings.items():
            if cluster_j != cluster_i:
                inter_sim = util.cos_sim(emb_i, emb_j)[0][0].item()
                inter_sims.append(inter_sim)

    if not intra_sims or not inter_sims:
        return 0.0

    sacc = np.mean(intra_sims) - np.mean(inter_sims)
    return (sacc + 1) / 2  # normalisé entre 0 et 1

def compute_semantic_coherence(df):
    """Calcule la cohérence sémantique intra-cluster basée sur les captions"""
    coherence_scores = []
    
    for cluster_id in df['cluster_id'].unique():
        cluster_data = df[df['cluster_id'] == cluster_id]
        if len(cluster_data) < 2:
            continue
            
        captions = [str(row['reference_text']) for _, row in cluster_data.iterrows() 
                   if pd.notna(row['reference_text']) and str(row['reference_text']).strip()]
        
        if len(captions) < 2:
            continue
            
        # Calculer la similarité moyenne intra-cluster
        embeddings = model.encode(captions)
        similarities = []
        
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = util.cos_sim(embeddings[i], embeddings[j])[0][0].item()
                similarities.append(sim)
        
        if similarities:
            coherence_scores.append(np.mean(similarities))
    
    return np.mean(coherence_scores) if coherence_scores else 0.0

# === FONCTION COMPLÈTE POUR CACC ET SACC ===
def plot_cacc_sacc_by_criterion(df_long):
    """
    Calcule et affiche les métriques CAcc et SAcc par critère
    selon la méthodologie du papier "Organizing Unstructured Image Collections using Natural Language"
    """
    criteria = df_long['criterion'].unique()
    cacc_scores = []
    sacc_scores = []
    sacc_pseudo_scores = []
    cluster_counts = []
    crit_labels = []

    print("\n=== ÉVALUATION CACC ET SACC ===")
    
    for crit in criteria:
        df_crit = df_long[df_long['criterion'] == crit]
        print(f"\n--- {crit.replace('_', ' ').title()} ---")
        print(f"Nombre de clusters: {df_crit['cluster_id'].nunique()}")
        print(f"Nombre d'images: {len(df_crit)}")
        
        # CAcc - Clustering Accuracy
        cacc = clustering_accuracy_cacc(df_crit['true_class'], df_crit['cluster_id'])
        
        # SAcc - Semantic Accuracy
        sacc = semantic_accuracy_sacc(df_crit)
        
        print(f"✅ CAcc (Clustering Accuracy): {cacc:.3f}")
        print(f"✅ SAcc (Semantic Accuracy): {sacc:.3f}")
        
        # Métriques additionnelles
        coherence = compute_semantic_coherence(df_crit)
        print(f"📊 Cohérence Sémantique: {coherence:.3f}")
        
        crit_labels.append(crit.replace('_', ' ').title())
        cacc_scores.append(cacc)
        sacc_scores.append(sacc)
        cluster_counts.append(df_crit['cluster_id'].nunique())

    # Visualisation
    x = np.arange(len(crit_labels))
    width = 0.25

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Graphique 1: CAcc
    bars1 = ax1.bar(x, cacc_scores, width, alpha=0.8, color='lightcoral', label='CAcc')
    ax1.set_ylabel('Clustering Accuracy (CAcc)')
    ax1.set_title('CAcc par Critère')
    ax1.set_xticks(x)
    ax1.set_xticklabels(crit_labels, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    ax1.legend()
    
    # Ajouter les valeurs sur les barres
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # Graphique 2: SAcc
    bars2 = ax2.bar(x, sacc_scores, width, alpha=0.8, color='skyblue', label='SAcc')
    ax2.set_ylabel('Semantic Accuracy (SAcc)')
    ax2.set_title('SAcc par Critère')
    ax2.set_xticks(x)
    ax2.set_xticklabels(crit_labels, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.legend()
    
    # Ajouter les valeurs sur les barres
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # Graphique 3: Comparaison CAcc vs SAcc
    ax3.bar(x - width/2, cacc_scores, width, alpha=0.8, color='lightcoral', label='CAcc')
    ax3.bar(x + width/2, sacc_scores, width, alpha=0.8, color='skyblue', label='SAcc')
    ax3.set_ylabel('Score')
    ax3.set_title('Comparaison CAcc vs SAcc')
    ax3.set_xticks(x)
    ax3.set_xticklabels(crit_labels, rotation=45, ha='right')
    ax3.set_ylim(0, 1)
    ax3.legend()
    
    # Graphique 4: Nombre de clusters
    bars4 = ax4.bar(x, cluster_counts, width, alpha=0.8, color='lightgreen')
    ax4.set_ylabel('Nombre de Clusters')
    ax4.set_title('Nombre de Clusters par Critère')
    ax4.set_xticks(x)
    ax4.set_xticklabels(crit_labels, rotation=45, ha='right')
    
    # Ajouter les valeurs sur les barres
    for bar in bars4:
        height = bar.get_height()
        ax4.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
    
    # Résumé des résultats
    print(f"\n=== RÉSUMÉ DES RÉSULTATS ===")
    print(f"CAcc moyen: {np.mean(cacc_scores):.3f} ± {np.std(cacc_scores):.3f}")
    print(f"SAcc moyen: {np.mean(sacc_scores):.3f} ± {np.std(sacc_scores):.3f}")
    print(f"Nombre moyen de clusters: {np.mean(cluster_counts):.1f}")
    
    return {
        'criteria': crit_labels,
        'cacc_scores': cacc_scores,
        'sacc_scores': sacc_scores,
        'cluster_counts': cluster_counts
    }