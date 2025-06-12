import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np


def draw_cluster_images_as_wordcloud(ax, image_paths, max_images=9):
    if not image_paths:
        ax.text(0.5, 0.5, 'Aucune image trouvée', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, color='red')
        return

    grid_size = 3
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(positions) >= min(max_images, len(image_paths)):
                break
            x = 0.05 + (i / grid_size) * 0.9
            y = 0.05 + (j / grid_size) * 0.9
            positions.append((x, y))

    for idx, img_path in enumerate(image_paths[:max_images]):
        if idx >= len(positions):
            break
        try:
            img = Image.open(img_path)
            size = 400
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            x, y = positions[idx]
            width = 1.0 / grid_size * 0.9
            height = 1.0 / grid_size * 0.9
            ax.imshow(img, extent=(x, x + width, y, y + height), aspect='auto', alpha=0.8)
        except Exception as e:
            print(f"Erreur chargement image {img_path}: {e}")
            continue

def display_all_clusters_on_one_figure(df_long, image_folder, max_images_per_cluster=9, save_dir=None):
    if not os.path.exists(image_folder):
        print(f"Erreur: Le dossier {image_folder} n'existe pas")
        return

    df_long = df_long.sort_values(by=['criterion', 'cluster_id'])
    grouped = df_long.groupby(['criterion', 'cluster_id'])
    criteria = df_long['criterion'].unique()
    clusters_per_criterion = {c: df_long[df_long['criterion'] == c]['cluster_id'].unique() for c in criteria}

    for criterion in criteria:
        clusters = clusters_per_criterion[criterion]
        n_clusters = len(clusters)
        n_cols = min(4, n_clusters)
        n_rows = int(np.ceil(n_clusters / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
        axes = axes.flatten() if n_clusters > 1 else [axes]
        for col_idx, cluster_id in enumerate(clusters):
            ax = axes[col_idx]
            try:
                df_cluster = grouped.get_group((criterion, cluster_id))
            except KeyError:
                continue
            cluster_str = str(cluster_id)
            ax.set_title(f"{cluster_str}\n({len(df_cluster)} images)", fontsize=10, pad=10, wrap=True)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            image_paths = []
            for _, row in df_cluster.iterrows():
                if row['image_path'] and os.path.exists(row['image_path']):
                    image_paths.append(row['image_path'])
                else:
                    image_name = row['image_name']
                    base_name = os.path.splitext(image_name)[0]
                    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                        for name_variant in [image_name, base_name + ext]:
                            test_path = os.path.join(image_folder, name_variant)
                            if os.path.exists(test_path):
                                image_paths.append(test_path)
                                break
                        else:
                            continue
                        break
            draw_cluster_images_as_wordcloud(ax, image_paths, max_images_per_cluster)
        # Désactive les axes inutilisés
        for j in range(len(clusters), len(axes)):
            axes[j].axis('off')
        fig.suptitle(criterion.replace('_', ' ').title(), fontsize=16)
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{criterion}_clusters.png"))
        plt.close(fig)

def display_clusters_per_criterion(df_long, image_folder, max_images_per_cluster=9):
    criteria = df_long['criterion'].unique()
    for criterion in criteria:
        clusters = df_long[df_long['criterion'] == criterion]['cluster_id'].unique()
        n_clusters = len(clusters)
        n_cols = min(4, n_clusters)
        n_rows = int(np.ceil(n_clusters / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
        axes = axes.flatten() if n_clusters > 1 else [axes]
        for i, cluster in enumerate(clusters):
            ax = axes[i]
            image_paths = df_long[(df_long['criterion'] == criterion) & (df_long['cluster_id'] == cluster)]['image_path'].tolist()
            ax.set_title(f"{str(cluster)[:25]}...", fontsize=10)
            draw_cluster_images_as_wordcloud(ax, image_paths, max_images=max_images_per_cluster)
        # Désactive les axes inutilisés
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        fig.suptitle(criterion.replace('_', ' ').title(), fontsize=16)
        plt.tight_layout()
        plt.show()