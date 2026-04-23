# Algo Tag Classifier

Prédiction automatique des tags d'exercices algorithmiques (Codeforces).

---

## Contexte

Le dataset est un sous-ensemble de [xCodeEval](https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval) composé de **4982 exercices** issus de Codeforces, chacun stocké dans un fichier JSON séparé.

**Objectif :** prédire les tags associés à un exercice parmi 8 tags cibles :

| Tag | Description |
|---|---|
| `math` | Problèmes mathématiques généraux |
| `graphs` | Théorie des graphes |
| `strings` | Manipulation de chaînes |
| `number theory` | Arithmétique, primalité, PGCD... |
| `trees` | Arbres (BFS/DFS sur arbre) |
| `geometry` | Géométrie computationnelle |
| `games` | Théorie des jeux |
| `probabilities` | Probabilités / espérance |

---

## Structure du projet

```
algo-classifier/
├── data/
│   ├── test_set.csv               # Test set baseline (20%)
│   └── test_set_optimized.csv     # Test set optimized (20%, même split)
├── models/
│   ├── baseline/                  # Modèle baseline
│   │   ├── classifier.pkl
│   │   ├── vectorizer.pkl
│   │   └── scaler.pkl
│   └── optimized/                 # Modèle optimisé
│       ├── classifier.pkl
│       ├── vectorizer.pkl
│       ├── scaler.pkl
│       ├── thresholds.pkl
│       ├── train_indices.npy
│       ├── test_indices.npy
│       └── embeddings_cache.npy   # Cache sentence-transformers
├── baseline/                      # Approche 1 : TF-IDF + LogisticRegression
│   ├── config.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── optimized/                     # Approche 2 : LightGBM + Embeddings + Seuils
│   ├── config.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── data_loader.py                 # Chargement partagé entre les deux approches
├── compare.py                     # Comparaison baseline vs optimized
├── requirements.txt
├── README.md
├── EXPLICATION.md
└── issues.md
```

---

## Installation

```bash
# 1. Créer et activer le venv
python -m venv venv
source venv/bin/activate

# 2. Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

### Approche Baseline

```bash
# Entraînement
python -m baseline.train

# Évaluation
python -m baseline.evaluate

# Prédiction
python -m baseline.predict --input data/sample.json
```

### Approche Optimisée

```bash
# Entraînement (génère les embeddings au premier run, ~10-20 min CPU)
python -m optimized.train

# Évaluation
python -m optimized.evaluate

# Prédiction
python -m optimized.predict --input data/sample.json
```

### Comparaison des deux approches

```bash
python compare.py
```

> ⚠️ Les deux modèles doivent être entraînés avant de lancer `compare.py`.

---

## Features utilisées

### Baseline

| Type | Détail | Dimensions |
|---|---|---|
| TF-IDF | Unigrammes + bigrammes sur texte concaténé | 20 000 |
| Features méta | Difficulté, time limit, longueurs de texte | 5 |
| Code patterns | Imports et mots-clés algorithmiques (binaires) | 42 |
| **Total** | | **20 047** |

### Optimisé

| Type | Détail | Dimensions |
|---|---|---|
| TF-IDF | Unigrammes + bigrammes sur texte concaténé | 20 000 |
| Features méta | Difficulté, time limit, longueurs de texte | 5 |
| Code patterns | Imports et mots-clés algorithmiques (binaires) | 51 (+9 number theory) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| **Total** | | **20 440** |

---

## Résultats

### Métriques globales

| Métrique | Baseline | Optimized | Delta |
|---|---|---|---|
| F1 micro | 0.617 | 0.630 | **+0.013** ✅ |
| F1 macro | 0.621 | 0.657 | **+0.036** ✅ |
| Hamming Loss | 0.0765 | 0.0621 | **-0.014** ✅ |

### Métriques par tag — Baseline

| Tag | Précision | Rappel | F1 | Support |
|---|---|---|---|---|
| `math` | 0.570 | 0.658 | 0.611 | 284 |
| `graphs` | 0.514 | 0.776 | 0.618 | 98 |
| `strings` | 0.600 | 0.890 | **0.717** | 91 |
| `number theory` | 0.415 | 0.603 | **0.492** | 73 |
| `trees` | 0.522 | 0.701 | 0.599 | 67 |
| `geometry` | 0.609 | 0.824 | 0.700 | 34 |
| `games` | 0.520 | 0.929 | 0.667 | 14 |
| `probabilities` | 0.441 | 0.789 | 0.566 | 19 |

### Métriques par tag — Optimized

| Tag | Précision | Rappel | F1 | Support | Delta |
|---|---|---|---|---|---|
| `math` | 0.577 | 0.644 | 0.609 | 284 | -0.002 |
| `graphs` | 0.647 | 0.561 | 0.601 | 98 | -0.017 |
| `strings` | 0.664 | 0.802 | **0.726** | 91 | +0.009 |
| `number theory` | 0.711 | 0.438 | **0.542** | 73 | **+0.050** |
| `trees` | 0.787 | 0.552 | 0.649 | 67 | +0.050 |
| `geometry` | 0.720 | 0.529 | 0.610 | 34 | -0.090 |
| `games` | 0.800 | 0.857 | **0.828** | 14 | **+0.161** 🔥 |
| `probabilities` | 0.846 | 0.579 | **0.688** | 19 | **+0.122** |

### Seuils de décision optimisés (par tag)

| Tag | Seuil | F1 obtenu |
|---|---|---|
| `math` | 0.142 | 0.609 |
| `graphs` | 0.100 | 0.601 |
| `strings` | 0.100 | 0.726 |
| `number theory` | 0.142 | 0.542 |
| `trees` | 0.100 | 0.649 |
| `geometry` | 0.100 | 0.610 |
| `games` | 0.100 | 0.828 |
| `probabilities` | 0.311 | 0.688 |

### Analyse

- 🔥 **Gains majeurs** : `games` (+0.161), `probabilities` (+0.122), `number theory` (+0.050)
- ✅ **F1 macro** : +0.036 — l'optimisation profite particulièrement aux tags rares
- ⚠️ **`geometry` régresse** (-0.090) — seuil trop bas (0.100) génère trop de faux positifs
- 📈 **Précision globalement améliorée** au profit du rappel sur certains tags

---

## Dépendances

```
pandas
numpy
scikit-learn
scipy
lightgbm
sentence-transformers
```