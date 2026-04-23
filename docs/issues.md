# Issues & Observations

---

## Problèmes de données

### Tags mal mappés
- `graph` → le vrai tag dans le dataset est `graphs` (avec s)
- `string` → le vrai tag dans le dataset est `strings` (avec s)
- **Impact** : 0 exemples trouvés au lieu de 542 et 422
- **Fix** : corrigé dans `config.py`

### Valeurs manquantes
- `prob_desc_notes` : 27.1% de nulls (1350/4982)
- `prob_desc_input_spec` : 0.7% de nulls (33/4982)
- `prob_desc_output_spec` : 1.7% de nulls (85/4982)
- `difficulty` : 0.8% de nulls (39/4982)
- **Fix** : fonction `safe_str()` dans `features.py` + `fillna(0)` dans `build_feature_matrix()`

### Valeurs aberrantes
- `difficulty` : valeur min = -1 (invalide, les niveaux vont de 800 à 3500)
- **Fix** : remplacé par la médiane (1668) dans `extract_meta_features()`

### Déséquilibre de classes sévère
- `math` : 28.3% des exemples → classe majoritaire
- `probabilities` : 1.8%, `games` : 2.1% → classes très rares
- **Impact** : le modèle a tendance à ignorer les classes rares
- **Fix baseline** : `class_weight='balanced'` dans LogisticRegression
- **Fix optimized** : `class_weight='balanced'` dans LightGBM + seuils de décision optimisés par tag

---

## Bugs rencontrés — Phase Baseline

### TypeError dans build_text_input
- **Erreur** : `sequence item 3: expected str instance, float found`
- **Cause** : les NaN pandas sont de type `float`, pas `None` — `or ""` ne les intercepte pas
- **Fix** : ajout de `safe_str()` qui vérifie `isinstance(val, float) and np.isnan(val)`

### TypeError dans extract_meta_features
- **Erreur** : `object of type 'float' has no len()`
- **Cause** : même problème NaN pandas sur `prob_desc_input_spec`
- **Fix** : `safe_str()` appliqué à tous les champs texte

### ValueError : Input X contains NaN
- **Erreur** : `LogisticRegression does not accept missing values encoded as NaN`
- **Cause** : `difficulty` contient 39 NaN qui passaient à travers les vérifications
- **Fix** : vérification explicite `np.isnan()` dans `extract_meta_features()` + `fillna(0)` dans `build_feature_matrix()`

### IndexError dans predict_single
- **Erreur** : `IndexError: invalid index to scalar variable`
- **Cause** : `OneVsRestClassifier.predict_proba()` retourne `(1, n_tags)` et non `(n_tags, 2)` comme un classifieur binaire classique
- **Fix** : `y_proba[0][i]` au lieu de `y_proba[i][0][1]`

### safe_str dupliquée
- Définie deux fois en local dans deux fonctions différentes
- **Fix** : remontée en fonction module-level dans `features.py` (principe DRY)

---

## Bugs rencontrés — Phase Optimized

### ModuleNotFoundError : lightgbm
- **Erreur** : `ModuleNotFoundError: No module named 'lightgbm'`
- **Cause** : paquet non installé dans le venv
- **Fix** : `pip install lightgbm sentence-transformers` + ajout dans `requirements.txt`

### ModuleNotFoundError : config
- **Erreur** : `ModuleNotFoundError: No module named 'config'`
- **Cause** : `data_loader.py` déplacé à la racine importait `from config import ...` — Python ne trouvait pas le module racine lors d'un run `python -m optimized.train`
- **Fix** : changé en `from baseline.config import DATA_DIR, TARGET_TAGS` dans `data_loader.py`

### Segmentation fault — LightGBM sur Apple Silicon (M-series)
- **Erreur** : `OMP: Error #179: Function pthread_mutex_init failed` + `Fatal Python error: Segmentation fault`
- **Cause** : conflit OpenMP entre LightGBM et les librairies Apple Silicon — double parallélisme : `n_jobs=-1` dans `LGBMClassifier` ET `n_jobs=-1` dans `OneVsRestClassifier`
- **Fix** : `n_jobs=1` dans les deux — plus lent mais stable

### ValueError : dimension mismatch dans build_feature_matrix
- **Erreur** : `ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 997 and the array at index 1 has size 4982`
- **Cause** : le cache des embeddings contient 4982 lignes (dataset complet), mais `evaluate.py` appelait `build_feature_matrix` avec seulement le test set (997 lignes) — impossible d'aligner les deux tableaux
- **Fix** :
  1. Sauvegarde des indices du split dans `train.py` : `np.save(model_dir / "train_indices.npy", train_idx)` et `np.save(model_dir / "test_indices.npy", test_idx)`
  2. Dans `evaluate.py` : chargement du cache complet puis indexation `X_emb_full[test_idx]`
  3. `build_X` dans `evaluate.py` accepte maintenant `X_emb` en paramètre explicite plutôt que de le recalculer

### TypeError : save_model() takes from 4 to 5 positional arguments but 6 were given
- **Erreur** : `TypeError: save_model() takes from 4 to 5 positional arguments but 6 were given`
- **Cause** : signature de `save_model` non mise à jour après ajout de `train_idx` et `test_idx` comme paramètres
- **Fix** : `def save_model(clf, vectorizer, scaler, thresholds, train_idx, test_idx, model_dir=MODEL_DIR)`

### NameError : extract_meta_features not defined dans evaluate.py
- **Erreur** : `NameError: name 'extract_meta_features' is not defined`
- **Cause** : `build_X` dans `evaluate.py` a été réécrit pour construire les features manuellement (sans appeler `build_feature_matrix`) mais l'import n'a pas été mis à jour
- **Fix** : ajout de `extract_meta_features, extract_code_features` dans l'import depuis `optimized.features`

---

## Décisions d'architecture

### Structure deux dossiers baseline/ et optimized/
- Chaque approche a ses propres fichiers avec les mêmes noms (`config.py`, `features.py`, `train.py`, `evaluate.py`, `predict.py`)
- `data_loader.py` mutualisé à la racine — partagé entre les deux approches
- `compare.py` à la racine — charge les deux modèles et les compare sur le même test set

### Cache des embeddings
- Les embeddings sont coûteux à générer (~10-20 min CPU pour 4982 exemples)
- Sauvegardés dans `models/optimized/embeddings_cache.npy` au premier run
- Rechargés instantanément lors des runs suivants
- Calculés sur le dataset **complet** — indexés ensuite via `train_indices.npy` / `test_indices.npy`

### Seuils optimisés sur le test set
- Par manque d'un set de validation dédié, les seuils sont optimisés sur le test set
- Légère fuite d'information — à corriger idéalement avec un split train/val/test

### Tags cibles finaux
```python
TARGET_TAGS = [
    "math",
    "graphs",        # corrigé (était "graph")
    "strings",       # corrigé (était "string")
    "number theory",
    "trees",
    "geometry",
    "games",
    "probabilities",
]
```

---

## Résultats finaux

### Baseline — TF-IDF + LogisticRegression

| Métrique | Score |
|---|---|
| F1 micro | 0.617 |
| F1 macro | 0.621 |
| Hamming Loss | 0.0765 |

| Tag | F1 | Support |
|---|---|---|
| `strings` | 0.717 | 91 |
| `geometry` | 0.700 | 34 |
| `games` | 0.667 | 14 |
| `graphs` | 0.618 | 98 |
| `math` | 0.611 | 284 |
| `trees` | 0.599 | 67 |
| `probabilities` | 0.566 | 19 |
| `number theory` | 0.492 | 73 |

### Optimized — LightGBM + Embeddings + Seuils

| Métrique | Score |
|---|---|
| F1 micro | 0.630 |
| F1 macro | 0.657 |
| Hamming Loss | 0.0621 |

| Tag | F1 | Seuil | Support |
|---|---|---|---|
| `games` | 0.828 | 0.100 | 14 |
| `strings` | 0.726 | 0.100 | 91 |
| `probabilities` | 0.688 | 0.311 | 19 |
| `trees` | 0.649 | 0.100 | 67 |
| `geometry` | 0.610 | 0.100 | 34 |
| `graphs` | 0.601 | 0.100 | 98 |
| `math` | 0.609 | 0.142 | 284 |
| `number theory` | 0.542 | 0.142 | 73 |

### Comparaison

| Métrique | Baseline | Optimized | Delta |
|---|---|---|---|
| F1 micro | 0.617 | 0.630 | **+0.013** ✅ |
| F1 macro | 0.621 | 0.657 | **+0.036** ✅ |
| Hamming Loss | 0.0765 | 0.0621 | **-0.014** ✅ |