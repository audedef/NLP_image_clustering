# X-Cluster: Reproducing Open-ended Semantic Clustering with MLLMs

This repository contains our implementation and evaluation of X-Cluster, a framework for unsupervised semantic multiple clustering of image collections using multimodal large language models (MLLMs). The work was conducted as part of the ML Reproducibility Challenge and reproduces the method presented in the paper:

**"Organizing Unstructured Image Collections using Natural Language"**
by Mingxuan Liu, Zhun Zhong, Jun Li, Gianni Franchi, Subhankar Roy, Elisa Ricci (2024).


## Project Overview

**X-Cluster** enables the automatic discovery of clustering criteria in natural language, and the subsequent grouping of images according to these criteria—without any human supervision. It performs:

* Caption generation via a multimodal LLM (e.g., LLaVA),
* Proposal and refinement of semantic clustering axes via a textual LLM (e.g., LLaMA),
* Multi-granularity image clustering (coarse / middle / fine),
* Evaluation of semantic coherence using Sentence-BERT embeddings.


## Implementation Details

We reproduce the **caption-based pipeline** of X-Cluster using:

* **MLLM:** [`ManishThota/llava_next_video`](https://ollama.com/library/llava-next-video)
* **LLM:** `llama3.1:8b` (via [Ollama](https://ollama.com))
* **Dataset:** [Food-101](https://data.vision.ee.ethz.ch/cvl/food-101/) (used to approximate Food-4c)

Due to hardware limitations, we process only **100 random images**, focusing on reproducibility over scale.


## Project Structure

* `script_clustering.py` – Main pipeline for captioning, proposing criteria, and clustering images.
* `cluster_utils.py` – Post-processing: normalizing labels, deduplication, data transformation.
* `food101_cluster_analysis.py` – Evaluation script computing CAcc and SAcc, generating visualizations.
* `metrics.py` – Metric implementations: Clustering Accuracy (CAcc), Semantic Accuracy (SAcc).
* `visualization.py` – Cluster visualization using PIL and matplotlib.

Output CSVs and cluster plots are generated for analysis.


## Evaluation Results

* **CAcc** (Clustering Accuracy): 0.09 – 0.18
* **SAcc** (Semantic Accuracy): 0.51 – 0.57
* Generated criteria include:
  `Color Scheme`, `Cooking Method`, `Composition Style`, `Meal Occasion`, etc.

These results confirm the model’s ability to organize image collections along interpretable, language-guided dimensions. Despite lower CAcc (expected due to unsupervised setup), the SAcc shows consistent semantic coherence.


## Reproduction Setup

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Start your Ollama server with required models:

```bash
ollama pull llama3:8b
ollama pull ManishThota/llava_next_video
```

### Folder Structure

Place Food-101 images in:

```
./data/food-101/images/
```

Then run the pipeline:

```bash
python script_clustering.py
python food101_cluster_analysis.py
```


## Limitations and Challenges

* Only 100 images were used (due to GPU constraints).
* Lighter models via Ollama (vs GPT-4/Claude used in the original paper).
* Some naming inconsistencies and cluster imbalance persist.


## Key Contributions

* Full reproduction of X-Cluster’s modular pipeline
* Semantic cluster discovery with no supervision
* Evaluation using visualizations and metrics


## Reference

If you use or build on this work, please cite:

> Mingxuan Liu, Zhun Zhong, Jun Li, Gianni Franchi, Subhankar Roy, Elisa Ricci (2024).
> "Organizing Unstructured Image Collections using Natural Language"
> [https://arxiv.org/abs/2404.02887](https://arxiv.org/abs/2404.02887)


## Contact

This reproduction was conducted by Aude De Fornel, Hugo De Gieter, Romain Donne, Camille Ishac, Yael Juarez-Martinez
As part of the ML Reproducibility Challenge 2025

