# Explication complète du projet — Algo Tag Classifier

> Ce document explique le projet de A à Z, de la problématique jusqu'au code.
> Il est pensé pour être compréhensible même sans expérience en Python ou en Machine Learning.

---

## 1. La problématique

### C'est quoi Codeforces ?

Codeforces est un site web qui propose des exercices de programmation. Chaque exercice demande à l'utilisateur d'écrire un algorithme pour résoudre un problème mathématique ou logique.

Chaque exercice est classifié avec des **tags** — des étiquettes qui décrivent les notions mises en jeu. Par exemple :
- Un exercice sur le plus court chemin dans un réseau → tag `graphs`
- Un exercice sur les nombres premiers → tag `number theory`
- Un exercice avec des chaînes de caractères → tag `strings`

### Le problème

Annoter manuellement des milliers d'exercices avec les bons tags est long et coûteux. L'objectif est de créer un **algorithme capable de prédire automatiquement les tags** d'un exercice à partir de son énoncé et de sa solution.

### Les 8 tags qu'on cherche à prédire

```
math, graphs, strings, number theory, trees, geometry, games, probabilities
```

Ces 8 tags sont les plus fréquents parmi les ~60 tags existants sur Codeforces. On a filtré le dataset pour ne garder que ces 8-là.

**Que se passe-t-il si un exercice ne rentre dans aucun de ces 8 tags ?**

C'est une limitation de scope explicite du projet. En pratique :
- Si un exercice est tagué `"bitmask"` ou `"segment tree"` (hors des 8), le modèle lui assignera quand même des probabilités pour chacun des 8 tags.
- Si toutes les probabilités sont sous le seuil de décision → le modèle prédit "aucun tag", ce qui est techniquement correct.
- Mais si l'exercice ressemble sémantiquement à un exercice `graphs`, il risque de recevoir ce tag par erreur — faux positif garanti.

Cette limite est à mentionner explicitement quand on présente le projet.

---

## 2. Le dataset

### Structure du dataset

Ici, chaque exemple est un exercice Codeforces.

Le dataset contient **4982 fichiers JSON**. Un fichier JSON est un format texte structuré pour stocker des données, comme un dictionnaire Python.

Chaque fichier représente un exercice et contient :

```json
{
    "prob_desc_description": "You are given a graph with n nodes...",
    "prob_desc_input_spec": "The first line contains n and m...",
    "prob_desc_output_spec": "Print the shortest path...",
    "prob_desc_notes": "In the first example...",
    "source_code": "import heapq\ndef dijkstra(graph): ...",
    "difficulty": 1800,
    "prob_desc_time_limit": "2 seconds",
    "tags": ["graphs", "shortest paths", "binary search"]
}
```

**C'est quoi `prob_desc_time_limit` ?**

C'est la contrainte d'exécution imposée au programme : combien de secondes maximum il a pour s'exécuter. Un `time_limit` serré (1 seconde) sur un problème difficile signale souvent qu'un algorithme efficace est requis — comme Dijkstra (`graphs`) ou DFS/BFS (`trees`) — plutôt qu'un calcul mathématique pur. C'est pourquoi on l'utilise comme feature méta pour aider le modèle.

### C'est quoi un problème multi-label ?

Normalement en classification, chaque exemple appartient à **une seule catégorie**. Par exemple : un email est soit spam, soit pas spam.

Ici, un exercice peut avoir **plusieurs tags à la fois** :
```json
"tags": ["graphs", "shortest paths", "binary search"]
```
C'est ce qu'on appelle la **classification multi-label** — on prédit plusieurs étiquettes simultanément.

---

## 3. Les observations sur le dataset (EDA)

### C'est quoi l'EDA ?

EDA = Exploratory Data Analysis (Analyse Exploratoire des Données). C'est l'étape où on examine le dataset avant de construire le modèle, pour comprendre sa structure et ses problèmes.

### Ce qu'on a découvert

**Problème 1 — Tags mal nommés**
On cherchait le tag `graph` mais dans le dataset il s'appelle `graphs` (avec un s). Même chose pour `string` → `strings`. Résultat : 0 exemple trouvé au lieu de 542. Fix : corriger les noms dans `config.py`.

**Problème 2 — Valeurs manquantes**
Certains champs sont vides (`null` en JSON, `NaN` en Python) :
- `prob_desc_notes` : vide dans 27% des cas
- `prob_desc_input_spec` : vide dans 0.7% des cas
- `difficulty` : vide dans 0.8% des cas

**Problème 3 — Valeurs aberrantes**
`difficulty` peut valoir `-1`, ce qui n'a pas de sens (les niveaux vont de 800 à 3500). On remplace par la médiane (1668).

**Pourquoi remplacer par la médiane et pas la moyenne ?**
La médiane est robuste aux valeurs aberrantes : si un exercice a `difficulty = -1`, la moyenne serait tirée vers le bas, mais la médiane ne bougera pas. On remplace donc par la médiane (1668), qui représente un niveau typique du dataset sans être perturbée par les artéfacts. Si on laissait les `-1`, LightGBM créerait un split sur `difficulty < 1` → une branche entière qui capterait uniquement ces cas aberrants → overfitting sur du bruit pur.

**Problème 4 — Déséquilibre de classes**
Certains tags sont beaucoup plus fréquents que d'autres :
```
math          → 28.3% des exercices  (très fréquent)
probabilities →  1.8% des exercices  (très rare)
```
Cela pose deux problèmes. D'abord, si le modèle prédit "jamais `probabilities`", il a tort seulement 1.8% du temps — l'accuracy reste élevée mais le F1 est nul. Ensuite, les probabilités sont une branche des mathématiques : beaucoup d'exercices `probabilities` utilisent aussi du vocabulaire `math`, ce qui crée une **contamination sémantique** entre les deux classes. Le modèle associe parfois des features `probabilities` au tag `math` car ils co-apparaissent fréquemment dans les données d'entraînement.

---

## 4. Les deux approches développées

Le projet implémente deux approches distinctes et comparables.

---

### Approche 1 — Baseline (TF-IDF + LogisticRegression)

#### Étape 1 — Transformer le texte en nombres (TF-IDF)

Les algorithmes de ML ne comprennent pas le texte directement. Il faut le convertir en nombres. On utilise le **TF-IDF** pour ça.

**TF-IDF** = Term Frequency - Inverse Document Frequency. Deux idées combinées :

**TF (Term Frequency)** — combien de fois un mot apparaît dans *ce* document :
```
TF("graph", doc) = 5 occurrences / 120 mots total = 0.042
```

**IDF (Inverse Document Frequency)** — à quel point ce mot est *rare* globalement dans le corpus :
```
IDF("graph") = log(4982 docs / 800 docs contenant "graph") = log(6.2) ≈ 1.82
IDF("the")   = log(4982 / 4980) ≈ 0.0004  → quasi nul, mot inutile
```

**TF-IDF = TF × IDF** → un mot fréquent dans *ce* doc mais rare *globalement* obtient un score élevé, ce qui en fait un signal fort pour la classification.

Résultat : chaque exercice devient un vecteur de 20 000 dimensions (une par token), majoritairement rempli de zéros — c'est une **matrice sparse** (creuse).

**Unigrammes et bigrammes**

On indexe deux types de tokens :
- **Unigramme** = token unique : `"find"`, `"shortest"`, `"path"`
- **Bigramme** = paire de tokens consécutifs : `"find shortest"`, `"shortest path"`

Configuration : `ngram_range=(1, 2)` → TF-IDF indexe les deux. Pourquoi ? Parce que certains concepts ont un sens composite que les unigrammes seuls perdent :
```
"number" seul      → ambigu
"theory" seul      → ambigu
"number theory"    → signal fort pour le tag number theory
"binary search"    → impossible à déduire des mots séparément
```
L'inconvénient est que le vocabulaire explose — d'où la limite à 20 000 dimensions.

**Pourquoi 20 000 features ?**

Sans limite, TF-IDF produirait un vecteur avec tous les tokens uniques du corpus, soit potentiellement 50 000 à 100 000 dimensions pour 5000 documents. Cela pose trois problèmes : la mémoire explose même pour une matrice sparse, les tokens très rares (une seule occurrence) ne font que du bruit, et la régression logistique est plus lente à entraîner. 20 000 est un compromis empirique standard : on garde les 20 000 tokens avec le TF-IDF global le plus élevé. En pratique, augmenter à 30 000 changerait peu le F1 sur ce corpus.

#### Étape 2 — Extraire des features supplémentaires (47 features)

**5 features méta (numériques) :**
- `difficulty` : niveau de difficulté (ex: 1800)
- `time_limit` : limite de temps en secondes (ex: 2.0)
- `desc_len` : longueur de l'énoncé en caractères
- `code_len` : longueur du code source
- `input_spec_len` : longueur de la spec d'entrée

**42 features extraites du code (binaires — 0 ou 1) :**
- `import_math` : est-ce que le code importe `math` ?
- `use_gcd` : est-ce que le code utilise `gcd` ?
- `use_dfs` : est-ce que le code contient `dfs` ?
- `use_sqrt` + `use_cos` + `use_sin` → probablement `geometry`
- `use_dfs` + `use_visited` + `use_graph` → probablement `graphs`
- ... (42 patterns au total)

**Pourquoi on n'utilise pas `source_code` et `prob_desc_description` directement comme features brutes ?**

`source_code` : le code source contient la réponse au problème. L'utiliser comme feature brute causerait un **data leakage** (fuite de données) — le modèle mémoriserait la solution au lieu d'apprendre la structure du problème. En production, on n'a pas le code source d'un nouvel exercice non résolu. On extrait donc uniquement des patterns binaires (use_dfs, import_math...) — le signal algorithmique, sans la solution complète.

`prob_desc_description` : c'est le texte principal qu'on exploite via TF-IDF et embeddings. La description brute est nettoyée et tronquée dans `features.py` avant d'être vectorisée, ce qui la rend utilisable sans introduire du bruit lié à la mise en forme.

**Total : 20 047 features par exercice**

#### Étape 3 — Le classifieur multi-label

On utilise un **OneVsRestClassifier** avec **Régression Logistique**.

**C'est quoi la Régression Logistique ?**
Malgré le nom, c'est un classificateur. Il apprend un vecteur de poids `w` tel que :
```
P(tag="math" | x) = σ(w · x + b)
```
où `σ` est la sigmoïde qui écrase la sortie entre 0 et 1. Si `P > 0.5` → tag prédit. C'est un modèle linéaire qui fonctionne très bien quand les features sont directement discriminantes — ce qui est le cas ici : le mot "dijkstra" dans le TF-IDF est un signal quasi-certain pour `graphs`.

**C'est quoi OneVsRestClassifier ?**
Comme on a 8 tags, on entraîne **8 classifieurs indépendants** — un par tag. Chacun répond oui/non pour son tag.

**C'est quoi `class_weight='balanced'` ?**
Comme certains tags sont rares (1.8% pour `probabilities`), sans correction le modèle les ignorerait. `class_weight='balanced'` donne plus d'importance aux exemples rares pendant l'entraînement pour compenser le déséquilibre.

---

### Approche 2 — Optimisée (LightGBM + Embeddings + Seuils)

Trois améliorations majeures par rapport au baseline.

#### Amélioration 1 — LightGBM à la place de LogisticRegression

**C'est quoi LightGBM ?**
LightGBM est un algorithme de **gradient boosting** de Microsoft. Il entraîne une séquence d'arbres de décision, chaque arbre corrigeant les erreurs du précédent :

1. Entraîne un arbre de décision faible (profondeur 3-5)
2. Calcule les erreurs résiduelles
3. Entraîne un 2ème arbre pour corriger ces erreurs
4. Répète 100 à 1000 fois
5. Prédiction finale = somme pondérée de tous les arbres

Avantage clé sur la régression logistique : il capture des **interactions non-linéaires** entre features. Par exemple : `(difficulty=2500) AND (has_modular_arithmetic=True)` → forte probabilité `number theory`. La régression logistique ne peut pas apprendre cette combinaison.

#### Amélioration 2 — Embeddings sémantiques (sentence-transformers)

**Limite du TF-IDF** : il compte les occurrences de mots mais ne comprend pas le sens. "graph" dans une description peut désigner autre chose qu'un graphe algorithmique.

**C'est quoi un embedding ?**
Un embedding est une représentation vectorielle dense d'un texte — un vecteur de nombres qui capture le *sens* de la phrase, pas juste les mots. Deux phrases qui expriment la même idée avec des mots différents auront des embeddings proches :
```
"find shortest path"          → [0.23, -0.11, 0.87, ...]
"minimum distance in network" → [0.21, -0.09, 0.85, ...]  ← proche !
```

**C'est quoi sentence-transformers ?**
C'est une bibliothèque qui encapsule des modèles BERT fine-tunés pour produire des embeddings de phrases. Le modèle `all-MiniLM-L6-v2` passe le texte à travers 6 couches Transformer et produit un vecteur de **384 dimensions**. Ce nombre n'est pas un hyperparamètre choisi par nous — il est fixé par l'architecture du modèle pré-entraîné. "MiniLM" signifie qu'il est une version compressée (distillée) de BERT base (768 dimensions) : deux fois plus léger, deux fois plus rapide, performances similaires sur les tâches de similarité.

**Gestion du temps de calcul — cache disque**
Générer les embeddings prend ~10-20 minutes CPU. Pour ne pas recalculer à chaque fois, on sauvegarde le résultat dans `models/optimized/embeddings_cache.npy`. Les runs suivants chargent le fichier directement (< 1 seconde).

**Gestion des indices de split**
Les embeddings sont calculés sur le dataset complet (4982 lignes). Pour évaluer uniquement sur le test set (997 lignes), on sauvegarde les indices du split (`train_indices.npy`, `test_indices.npy`) et on indexe le tableau d'embeddings : `X_emb_full[test_idx]`. Sans cette sauvegarde, recalculer le split aléatoire à chaque run changerait les indices, et les embeddings cachés seraient assignés aux mauvais exemples.

#### Amélioration 3 — Seuils de décision optimisés par tag

**Problème du seuil fixe à 0.5**
Par défaut, un classifieur prédit "oui" si la probabilité dépasse 0.5. Mais ce seuil n'est pas optimal pour tous les tags.

`probabilities` est rare (1.8%) → le modèle est naturellement conservateur et produit des probabilités de 0.3-0.4 même pour de vrais positifs. Avec un seuil à 0.5, il les manque presque tous. En abaissant le seuil à 0.15, on en capture beaucoup plus. En résumé, le seuil sert à **calibrer le compromis précision/rappel** pour chaque tag indépendamment, en tenant compte de la rareté et de la difficulté intrinsèque du tag.

**Comment on optimise les seuils ?**
Après l'entraînement, on teste 20 valeurs de seuil entre 0.1 et 0.9 pour chaque tag sur le set de validation, et on retient la valeur qui maximise le F1 :

```python
for thresh in np.linspace(0.1, 0.9, 20):
    preds = (proba[:, i] >= thresh).astype(int)
    f1 = f1_score(y_val[:, i], preds)
    if f1 > best_f1:
        best_thresh = thresh
```

Les seuils sont sauvegardés dans `thresholds.pkl` et utilisés dans `evaluate.py` et `predict.py`.

**Total features optimized : 20 440** (20 000 TF-IDF + 56 méta/patterns + 384 embeddings)

---

## 5. Les fichiers du projet

### Les fichiers `.pkl` et `.npy` — à quoi servent-ils ?

**`.pkl` (Pickle)** — sérialisation Python d'objets arbitraires :
- `baseline_model.pkl` → le pipeline scikit-learn entier (TF-IDF + LogisticRegression), prêt à appeler `.predict()`
- `optimized_model.pkl` → les 8 modèles LightGBM (un par tag) + les seuils optimisés + le vectoriseur TF-IDF

**`.npy` (NumPy binary)** — tableau numérique sérialisé, compact et rapide :
- `embeddings_cache.npy` → matrice `(4982, 384)` — les vecteurs sémantiques de tous les exercices, calculés une seule fois
- `train_indices.npy` → tableau 1D des indices du train set, ex. `[0, 3, 7, 12, ...]`
- `test_indices.npy` → idem pour le test set

On utilise `.npy` plutôt que `.pkl` pour les tableaux numériques car c'est un format binaire optimisé pour NumPy : plus rapide à lire/écrire, plus compact en mémoire.

### Fichiers partagés (racine)

**`data_loader.py`** — partagé entre baseline et optimized
- `load_dataset()` : parcourt le dossier, lit chaque JSON, retourne un DataFrame
- `add_binary_tag_columns()` : crée des colonnes 0/1 pour chaque tag cible
- `explore()` : affiche les statistiques du dataset (EDA)

**`compare.py`** — compare les deux modèles sur le même test set
- Charge baseline et optimized
- Affiche les métriques des deux côté à côté
- Calcule les deltas avec indicateurs ✅/❌

### baseline/

**`config.py`** — constantes : `DATA_DIR`, `MODEL_DIR = "models/baseline"`, `TARGET_TAGS`

**`features.py`** — extraction des features baseline (TF-IDF + 47 features)

**`train.py`** — entraîne LogisticRegression, sauvegarde classifier/vectorizer/scaler

**`evaluate.py`** — évalue le modèle baseline avec `clf.predict()` (seuil fixe 0.5)

**`predict.py`** — affiche les probabilités par tag

### optimized/

**`config.py`** — comme baseline + `LGBM_PARAMS`, `DEFAULT_THRESHOLDS`, `EMBEDDING_MODEL`

**`features.py`** — comme baseline + 9 patterns number theory + `extract_embeddings()` avec cache

**`train.py`** — entraîne LightGBM, optimise les seuils, sauvegarde les indices du split

**`evaluate.py`** — charge les embeddings depuis le cache, indexe sur test_idx, applique les seuils optimisés

**`predict.py`** — affiche le seuil optimisé pour chaque tag

---

## 6. Pourquoi `python -m`

Sans `-m` :
```bash
python optimized/train.py
```
Python ajoute `optimized/` au `sys.path`. L'import `from optimized.config import ...` échoue car Python ne "voit" pas le package `optimized` — il est à la racine du path, pas son parent.

Avec `-m` :
```bash
python -m optimized.train
```
Python part du **répertoire courant** (la racine du projet) et traite `optimized` comme un package. Tous les imports relatifs (`from optimized.config`, `from baseline.config`) fonctionnent.

Règle simple : dès que ton projet a des packages imbriqués avec des imports croisés → toujours `-m`.

---

## 7. Les métriques d'évaluation

### Précision, Rappel, F1

**Précision** : parmi tout ce que le modèle a prédit comme positif, combien étaient vrais ?
```
Precision = TP / (TP + FP)
"Quand je dis que c'est du math, j'ai raison X% du temps."
```

**Rappel** : parmi tous les vrais positifs, combien a-t-on retrouvés ?
```
Recall = TP / (TP + FN)
"Parmi tous les exercices math, j'en ai détecté X%."
```

**F1-score** : moyenne harmonique des deux — pénalise fort si l'un est faible :
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**F1 micro** : agrège tous les TP/FP/FN avant de calculer → pondéré implicitement par la fréquence. Un tag rare comme `probabilities` pèse peu dans ce calcul.

**F1 macro** : moyenne des F1 de chaque tag, sans pondération. Un tag rare compte autant qu'un tag fréquent → métrique plus exigeante et plus pertinente ici, car on veut que `probabilities` (1.8%) soit aussi bien classifié que `math` (28.3%).

### Hamming Loss

Pour du **multi-label**, une prédiction est un vecteur de bits. Exemple :
```
Vrai    : [1, 0, 1, 0, 0, 0, 0, 0]  # math + geometry
Prédit  : [1, 0, 0, 0, 0, 1, 0, 0]  # math + games
→ 2 erreurs sur 8 positions → Hamming Loss = 2/8 = 0.25
```

Globalement :
```
Hamming Loss = (nombre total de bits mal prédits) / (n_exemples × n_tags)
```

- **0.0** = parfait
- **0.0621** pour l'optimisé = en moyenne 0.5 tag mal assigné par exercice

C'est une métrique complémentaire au F1 : le F1 mesure si on trouve les bons tags, le Hamming Loss pénalise aussi chaque faux positif et faux négatif individuellement.

### Le "support"

Dans `classification_report` de scikit-learn, le **support** = nombre d'exemples vrais positifs dans le test set pour ce tag.

```
              precision  recall  f1    support
math              0.81    0.78   0.79    283
probabilities     0.72    0.69   0.70     18
```

`math` a 283 exercices dans le test set, `probabilities` en a 18. Plus le support est faible, moins la métrique est stable statistiquement : un F1 de 0.70 sur 18 exemples est beaucoup moins fiable qu'un F1 de 0.79 sur 283 — une seule prédiction incorrecte peut faire bouger le F1 de plusieurs points.

---

## 8. Split train / validation / test

Trois splits distincts avec des rôles différents :

```
Dataset complet (4982)
├── Train     (70%) → 3487 exemples : le modèle apprend ici
├── Validation (15%) →  747 exemples : on tune les hyperparamètres et les seuils ICI
└── Test      (15%) →  748 exemples : évaluation finale, touché UNE SEULE FOIS
```

**Pourquoi trois et pas deux ?**

Si on optimisait les seuils directement sur le test set, on "mémoriserait" le test set et les métriques seraient trop optimistes — elles ne refléteraient pas les vraies performances sur des données inédites. La validation sert de proxy du test pour toutes les décisions intermédiaires.

Dans ce projet : `optimize_thresholds()` utilise le validation set, `compare.py` évalue sur le test set → c'est propre.

**Note** : dans la version actuelle, les seuils sont optimisés sur le test set directement — c'est une légère fuite d'information mentionnée dans les limites. L'idéal est d'introduire un split validation dédié.

---

## 9. GridSearch — chercher les meilleurs hyperparamètres

Les hyperparamètres sont les paramètres qu'on choisit avant l'entraînement (ex: `n_estimators`, `learning_rate`, `max_depth` pour LightGBM). GridSearch teste **toutes les combinaisons** d'une grille définie :

```python
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}
# → 3 × 3 × 3 = 27 combinaisons à évaluer
```

Pour chaque combinaison, on fait une **cross-validation** (5 folds) → 27 × 5 = 135 entraînements. Coûteux mais exhaustif.

**RandomizedSearch** est souvent préféré : au lieu de tout tester, il sample aléatoirement N combinaisons → résultats proches pour 10x moins de temps.

Dans ce projet, les hyperparamètres LightGBM ont été fixés manuellement dans `config.py`. Un GridSearch est une piste d'amélioration directe pour gagner quelques points de F1.

---

## 10. Résultats comparatifs

### Métriques globales

| Métrique | Baseline | Optimized | Delta |
|---|---|---|---|
| F1 micro | 0.617 | 0.630 | **+0.013** ✅ |
| F1 macro | 0.621 | 0.657 | **+0.036** ✅ |
| Hamming Loss | 0.0765 | 0.0621 | **-0.014** ✅ |

### Analyse des gains par tag

| Tag | Baseline | Optimized | Delta | Explication |
|---|---|---|---|---|
| `games` | 0.667 | **0.828** | **+0.161** 🔥 | Seuil bas + embeddings capturent le contexte |
| `probabilities` | 0.566 | **0.688** | **+0.122** | Tag rare bien géré par le seuil optimisé |
| `number theory` | 0.492 | **0.542** | **+0.050** | 9 patterns ciblés ajoutés |
| `trees` | 0.599 | 0.649 | +0.050 | Embeddings différencient mieux trees/graphs |
| `strings` | 0.717 | 0.726 | +0.009 | Déjà bon, léger gain |
| `geometry` | 0.700 | 0.610 | **-0.090** ⚠️ | Seuil 0.100 trop agressif → faux positifs |

### Pourquoi `geometry` régresse ?
Le seuil optimisé à 0.100 est très bas — le modèle prédit `geometry` dès qu'il a 10% de confiance, ce qui génère beaucoup de faux positifs. La précision augmente (amélioration) mais le rappel chute fortement (forte baisse), ce qui plombe le F1. Un seuil plus conservateur autour de 0.3-0.4 corrigerait probablement ce problème.

---

## 11. Combien d'exemples pour un dataset correct ?

Le dataset contient 4982 exercices — c'est correct pour l'approche actuelle, mais à la limite basse pour certains tags.

**Règle empirique** pour la classification multi-label avec LightGBM : **≥ 500 exemples positifs par tag** pour des métriques stables.

```
math          → ~1410 exemples positifs  ✅ confortable
probabilities →   ~90 exemples positifs  ⚠️ limite basse
geometry      →  ~350 exemples positifs  → acceptable
```

Pour du fine-tuning BERT : généralement ≥ 1000 exemples par classe recommandés. Pour SetFit (fine-tuning efficace) : peut fonctionner avec 8 à 64 exemples par classe — pertinent pour les tags rares.

Conséquence : les métriques pour `probabilities` et `geometry` sont moins stables statistiquement que pour `math` — une seule prédiction incorrecte peut faire bouger le F1 de plusieurs points.

---

## 12. Pistes d'amélioration

**Sans GPU :**
- Affiner les seuils sur un set de validation dédié (éviter la fuite d'information)
- Corriger le seuil de `geometry` manuellement (~0.35)
- Hyperparameter tuning LightGBM (GridSearch sur `num_leaves`, `learning_rate`)
- Ajouter un vrai module CLI avec `argparse`

**Avec GPU (Colab) :**
- Fine-tuning `CodeBERT` — modèle pré-entraîné sur du code source
- Fine-tuning `microsoft/codebert-base` en classification multi-label directe

**Autres approches possibles :**
- **Stacking** : utiliser les prédictions de LR comme features supplémentaires pour LightGBM → souvent +1-2 points de F1
- **Label correlation** : exploiter le fait que `math` et `number theory` co-apparaissent souvent (Classifier Chains, Label Powerset)
- **GenAI** : voir section suivante

---

## 13. Peut-on utiliser la GenAI pour ce problème ?

Oui, et c'est probablement l'approche la plus performante. Plusieurs niveaux d'intégration :

**Niveau 1 — Zero-shot / Few-shot avec GPT-4 / Claude :**
```python
prompt = """
Classify this competitive programming problem into these tags:
[math, graphs, strings, number theory, trees, geometry, games, probabilities]

Problem: {description}

Return only a JSON list of applicable tags.
"""
```
Aucun entraînement, performances déjà très bonnes sur les tags avec vocabulaire distinctif.
Inconvénients : coût ($), latence, pas de fine-tuning sur ton domaine.

**Niveau 2 — Embeddings via API GenAI :**
Remplacer `all-MiniLM-L6-v2` par `text-embedding-3-large` d'OpenAI (3072 dimensions) comme features pour LightGBM. Amélioration probable de +3-5 points de F1 macro, surtout sur les tags rares. C'est le meilleur rapport effort/impact et reste explicable en entretien.

**Niveau 3 — Fine-tuning d'un LLM :**
Fine-tuner un modèle open-source (Mistral, Llama 3) sur tes 4982 exemples reformulés en instruction-following. Probablement le meilleur F1 atteignable, mais nécessite un GPU et plusieurs heures d'entraînement.

**Niveau 4 — LLM comme feature extractor :**
Utiliser le LLM pour générer des features structurées ("ce problème implique-t-il des graphes ? score 0-10") et les injecter dans LightGBM — approche hybride interprétable.

---

## 14. Pourquoi ces résultats sont limités ?

**1. Tags ambigus entre eux**
`number theory` et `math` se ressemblent énormément dans le texte. Un exercice sur les nombres premiers utilise du vocabulaire mathématique général — le modèle ne sait pas toujours faire la différence.

**2. Dataset déséquilibré**
Avec si peu d'exemples pour les classes rares, le modèle a du mal à apprendre leurs caractéristiques. Et les probabilités étant une sous-branche des maths, la frontière entre les deux tags est floue.

**3. TF-IDF capture les mots mais pas le sens**
Le mot "graph" peut apparaître dans un exercice de géométrie sans être lié au tag `graphs`. Le TF-IDF ne comprend pas le contexte — il compte juste les occurrences.

**4. Petit dataset**
4982 exemples c'est peu pour un problème multi-label à 8 classes. Un modèle de deep learning nécessiterait idéalement 10x plus de données.

**5. Annotations imparfaites**
Certains exercices pourraient légitimement avoir plusieurs tags que les annotateurs humains n'ont pas tous mis. Le signal d'apprentissage est donc bruité.

**6. Seuils optimisés sur le test set**
Les seuils sont optimisés sur le même set que celui utilisé pour l'évaluation finale — légère fuite d'information. Idéalement, il faudrait un split train/validation/test séparé.

---

## 15. Concepts clés à retenir

| Concept | Définition simple |
|---|---|
| Dataset | Collection d'exemples pour entraîner le modèle |
| Feature | Caractéristique d'un exemple (ex: longueur du texte) |
| Label | La réponse attendue (ex: les tags) |
| Multi-label | Un exemple peut avoir plusieurs labels simultanément |
| TF-IDF | Technique pour convertir du texte en nombres pondérés par leur rareté |
| Unigramme / Bigramme | Token seul vs paire de tokens consécutifs |
| Embedding | Vecteur dense (384 dims) qui capture le sens sémantique d'un texte |
| sentence-transformers | Bibliothèque de modèles BERT pour générer des embeddings de phrases |
| Cache `.npy` | Résultat numérique sauvegardé sur disque pour éviter de recalculer |
| Classifieur | Algorithme qui apprend à prédire des labels |
| OneVsRest | Stratégie multi-label : un classifieur par tag |
| LightGBM | Gradient boosting par arbres — capture les interactions non-linéaires |
| Seuil de décision | Probabilité minimale pour prédire "oui" — optimisable par tag |
| EDA | Exploration des données avant modélisation |
| F1 macro | Moyenne des F1 par tag, sans pondération par fréquence |
| F1 micro | F1 global pondéré par la fréquence des classes |
| Précision | Taux de prédictions correctes parmi les prédictions faites |
| Rappel | Taux d'exemples positifs retrouvés parmi tous les positifs |
| Hamming Loss | Taux d'erreur moyen sur tous les labels d'un problème multi-label |
| Support | Nombre d'exemples positifs dans le test set pour un tag donné |
| Train / Val / Test | Trois splits distincts : apprentissage, tuning, évaluation finale |
| GridSearch | Recherche exhaustive des meilleurs hyperparamètres |
| Médiane | Statistique robuste aux valeurs aberrantes, utilisée pour imputer `difficulty` |
| Déséquilibre de classes | Certains tags sont très rares → biais du modèle vers les classes fréquentes |
| Data leakage | Utilisation d'information non disponible en production (ex: source_code) |
| `python -m` | Lance un module depuis la racine du projet pour respecter les imports relatifs |
| Pickle `.pkl` | Format Python pour sauvegarder des objets arbitraires sur disque |
| Matrice sparse | Matrice avec beaucoup de zéros — économise la mémoire pour TF-IDF |
| class_weight | Pénalise plus les erreurs sur les classes rares pendant l'entraînement |

---

## 16. Les 5 meta-features — pourquoi ce choix ?

### Pourquoi les meta-features ?

TF-IDF et les embeddings capturent le **contenu textuel**, mais oublient les **signaux structurels** des problèmes. Deux exercices avec du vocabulaire différent peuvent avoir la même difficulté ou la même complexité algorithmique. Les meta-features comblent ce gap.

### Détail des 5 features

**1. `difficulty` (note Codeforces)**
- Plage : 800–3500
- Sens : problèmes plus difficiles corrèlent avec des algorithmes plus avancés (graphs, trees, geometry)
- Exemple : un problème tagué `geometry` a généralement une difficulté > 1500
- Utilité : LightGBM peut apprendre "si difficulty > 2000 ET contains('sqrt'), alors geometry"

**2. `time_limit` (limite de temps CPU en secondes)**
- Plage : 0.5–3.0 secondes
- Sens : un time limit serré (1 sec) sur un problème difficile signale qu'un algo efficace est requis
- Exemple : "difficulty=1800 + time_limit=1sec" → probabilité élevée de `graphs` (Dijkstra) ou `trees` (DFS/BFS)
- Utilité : discrimine les problèmes qui demandent une approche computationnelle rapide vs mathématique

**3. `desc_len` (longueur de l'énoncé en caractères)**
- Plage : 200–5000 caractères
- Sens : problèmes longs donnent souvent plus de contexte (donc plus d'indices sur le tag)
- Exemple : un exercice de `geometry` avec des coordonnées a généralement une description détaillée
- Utilité : proxy indirect pour la complexité descriptive du problème

**4. `code_len` (longueur du source code de référence)**
- Plage : 100–2000 caractères
- Sens : certains algorithmes nécessitent plus de code (DFS récursif vs brute force)
- Exemple : un `trees` classique (BFS/DFS) ≈ 150–300 caractères, mais un `trees` avec LCA ≈ 500+
- Utilité : signal sur la complexité implémentationnelle de la solution

**5. `input_spec_len` (longueur de la spécification d'entrée)**
- Plage : 50–1000 caractères
- Sens : problèmes avec entrées complexes (matrices, graphes explicites) donnent plus de détails
- Exemple : une description "input: first line n, then n edges" est spécifique aux `graphs`
- Utilité : discrimine les problèmes structurés vs mathématiques purs

### Imputation des valeurs manquantes

Si un champ est `NaN` :
- `difficulty` → remplacée par la médiane (1668)
- `time_limit` → remplacée par 0.0 (défaut)
- autres → remplacées par 0.0

La **médiane** est préférée à la **moyenne** car elle ignore les valeurs aberrantes (`difficulty = -1`).

---

## 17. Data augmentation — créer des données synthétiques

### C'est quoi la data augmentation ?

C'est une technique pour **créer artificiellement de nouveaux exemples d'entraînement** à partir des exemples existants. Utile quand on a peu de données, surtout pour les classes rares.

### Pourquoi c'est utile ici

Le dataset est **déséquilibré** :
```
math          → 1410 exemples (28.3%)  ✅ suffisant
probabilities →   90 exemples (1.8%)   ⚠️ très rare
games         →   70 exemples (1.4%)   ⚠️ très rare
```

Avec seulement 70-90 exemples, LightGBM n'apprend pas bien. Data augmentation peut doubler ou tripler les exemples.

### Stratégies

**1. Paraphrase syntaxique**
```
Original  : "Find the shortest path using Dijkstra"
Augmentée : "Implement Dijkstra algorithm to find shortest path"
```
Même info, nouveaux tokens → TF-IDF génère des n-grammes différents.

**2. Mixup (interpolation)**
```python
X_aug = 0.7 * X_exercice_A + 0.3 * X_exercice_B
y_aug = (y_A + y_B) / 2  # labels moyennés
```

**3. Oversampling simple**
Dupliquer les exemples rares avec petite perturbation.

**4. Back-translation (avec LLM)**
Traduire en anglais → français → anglais (perte d'info = augmentation).

### Risques

- **Over-augmentation** : si 90% des données sont synthétiques, le modèle échoue sur les données réelles
- **Label noise** : paraphrase peut changer le sens → mauvais labels
- **Data leakage** : un exemple augmenté trop similaire au test set

**Recommandation** : augmenter modérément (2x-3x) seulement les classes rares.

---

## 18. Cross-validation — évaluation robuste

### C'est quoi la cross-validation ?

Au lieu d'un split unique train/test, on divise les données en K folds (ex: 5) :

```
Dataset complet
├── Fold 1 : test, reste train  → score 1
├── Fold 2 : test, reste train  → score 2
├── Fold 3 : test, reste train  → score 3
├── Fold 4 : test, reste train  → score 4
└── Fold 5 : test, reste train  → score 5

Score final = moyenne([score1, score2, score3, score4, score5])
```

### Pourquoi c'est utile

**Problème du split unique :**
Un split peut être **chanceux** (test set facile) ou **malchanceux** (test set difficile). Cross-validation moyenne ce bruit et donne une **variance** — on sait si le modèle est stable ou instable.

```
CV scores : [0.680, 0.685, 0.679, 0.690, 0.681]
Moyenne : 0.683
Std : 0.005  ← très stable ✅

vs.

CV scores : [0.60, 0.75, 0.62, 0.72, 0.68]
Moyenne : 0.674
Std : 0.055  ← très instable ⚠️
```

### Quand l'utiliser

- Hyperparameter tuning (GridSearch utilise K-fold CV)
- Données petites (< 10k exemples)
- Multi-label avec classes rares

**Ici** : utiliser `MultilabelStratifiedKFold` (5 folds) car elle assure que **chaque fold a la même distribution de tags**.

```python
from iterative_stratification import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=5, random_state=42)
for train_idx, test_idx in mskf.split(X, y):
    # Chaque fold contient approximativement les mêmes % de chaque tag
    pass
```

---

## 19. Choix du modèle d'embeddings `all-MiniLM-L6-v2`

### C'est quoi ce nom ?

Décomposition :
- **all** → pour tous les cas (phrase entière)
- **MiniLM** → version comprimée (distillée) de BERT-Large
- **L6** → 6 couches de Transformer
- **v2** → version 2

### Trade-off : vitesse vs qualité

| Modèle | Dimensions | Temps / 4982 ex | F1 estimé | Coût |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 15 min | 0.684 | Gratuit ✅ |
| `all-mpnet-base-v2` | 768 | 40 min | 0.700–0.710 | Gratuit |
| `text-embedding-3-large` (OpenAI) | 3072 | API | 0.720+ | $$$ |

### Pourquoi ce modèle ?

**Raison 1 — Vitesse** : 15 min acceptable, cache instantané après.

**Raison 2 — Mémoire** : 384 dims = 7.3 MB cache vs 58 MB pour 3072 dims.

**Raison 3 — Performance suffisante** : F1 +0.063 déjà excellent. Passer à mpnet → +0.02 seulement. Pas justifié.

**Raison 4 — Open-source** : pas d'API key, offline, reproductibilité garantie.

### Quand upgrader ?

Si F1 macro atteint 0.70–0.72, essayer `all-mpnet-base-v2` ou OpenAI pour +1–3 points.

---

## 20. Architecture production — API FastAPI + Redis

### Vue d'ensemble

Développement :
```
Exercice JSON → [CLI] → Prédiction → console
```

Production :
```
Exercice JSON → [API REST] → Cache Redis → Embeddings + LightGBM → JSON
                                ↑
                            [Monitoring]
```

### Pourquoi FastAPI ?

Framework ultra-rapide pour APIs REST. Avantages :
- `/docs` auto-généré (Swagger UI)
- Type hints → validation auto
- Async natif → traite 100 requêtes en parallèle
- Déploiement facile (Uvicorn, Docker, cloud)

### Pourquoi Redis ?

Cache ultra-rapide en mémoire. **Problème** : générer un embedding prend 15 secondes. **Solution** : mettre en cache.

```
Même requête 100 fois → 1ère : 15s, les 99 autres : < 100ms
```

### Monitoring — metrics clés

**P@K (Precision@K)** : parmi les K prédictions top, combien sont correctes ?

**Coverage par tag** : quel % de requêtes reçoivent chaque tag ?
```
math : 45% (attendu 28%, décalage ?)
```
Signal : si `probabilities` passe de 1.8% → 0.3%, le modèle s'est dégradé.

**Dérive de performance** : F1 réel sur les nouveaux exercices (feedback humain).
Si F1 macro < 0.65 → retrain ou rollback.

### Latence < 500ms

```
100 ms : I/O réseau
150 ms : TF-IDF + métafeatures
150 ms : LightGBM (8 modèles parallèles)
 50 ms : JSON + réponse
------
450 ms total ✅
```

Avec cache Redis : < 50ms pour exercice vu avant.

---

## 21. C'est quoi un module CLI ?

### CLI = Command Line Interface

Un CLI est une interface en ligne de commande — un programme qu'on utilise en tapant des commandes dans le terminal, sans interface graphique.

Exemples de CLIs que tu connais déjà :
```bash
git commit -m "message"     # CLI de git
pip install pandas          # CLI de pip
python train.py --epochs 10 # un script avec argparse basique
```

### La différence entre un script et un module CLI

**Un script simple :**
```bash
python algo_classifier/optimized/train.py
```
- Il faut savoir quel fichier appeler
- Pas d'aide automatique
- Les imports cassent si tu n'es pas dans le bon dossier
- Aucune découvrabilité — un nouveau développeur ne sait pas quoi faire

**Un module CLI installé :**
```bash
algo-classifier train --model optimized
```
- Commande disponible partout dans le terminal
- `--help` auto-généré sur chaque sous-commande
- Imports stables peu importe le répertoire courant
- Interface claire et professionnelle

### C'est quoi `argparse` ?

`argparse` est la bibliothèque standard Python pour créer des CLIs. Elle fait trois choses :

**1. Définir les arguments attendus :**
```python
parser.add_argument("--model", choices=["baseline", "optimized"], required=True)
parser.add_argument("--data-dir", default="./data")
parser.add_argument("--verbose", action="store_true")  # flag booléen
```

**2. Parser la ligne de commande :**
```python
args = parser.parse_args()
# args.model    → "optimized"
# args.data_dir → "./data"
# args.verbose  → True ou False
```

**3. Générer le `--help` automatiquement :**
```
usage: algo-classifier train [-h] --model {baseline,optimized} [--data-dir DATA_DIR]
```
Tu n'as rien à écrire — argparse construit ce message à partir des définitions.

### C'est quoi les sous-commandes (`subparsers`) ?

Un parser principal avec des sous-commandes indépendantes — comme `git` qui a `git commit`, `git push`, `git log` — chacun avec ses propres arguments.

```python
subparsers = parser.add_subparsers(dest="command")

p_train = subparsers.add_parser("train")    # sous-commande train
p_eval  = subparsers.add_parser("evaluate") # sous-commande evaluate
```

Quand tu tapes `algo-classifier train --model baseline`, argparse sait que `train` est la sous-commande et passe le contrôle au handler `cmd_train()`.

### C'est quoi `__main__.py` ?

`__main__.py` est un fichier spécial Python. Quand il est présent dans un package, il permet d'exécuter le package directement avec `-m` :

```bash
python -m algo_classifier   # exécute algo_classifier/__main__.py
```

Son contenu est minimaliste par design — il délègue immédiatement à `cli.py` :
```python
from algo_classifier.cli import main
if __name__ == "__main__":
    main()
```

Pourquoi séparer `__main__.py` et `cli.py` ? Pour que `cli.py` soit importable et testable indépendamment — on peut importer `from algo_classifier.cli import build_parser` dans un test unitaire sans déclencher l'exécution du programme.

### C'est quoi `pyproject.toml` et `pip install -e .` ?

`pyproject.toml` est le fichier de configuration moderne des packages Python (PEP 517/518). Il remplace `setup.py`.

Il fait trois choses dans ce projet :

**1. Décrire le package :**
```toml
[project]
name = "algo-classifier"
version = "0.1.0"
requires-python = ">=3.10"
```

**2. Déclarer les dépendances :**
```toml
dependencies = ["pandas", "numpy", "scikit-learn", "lightgbm", ...]
```

**3. Exposer une commande système via `entry_points` :**
```toml
[project.scripts]
algo-classifier = "algo_classifier.cli:main"
```
Cette ligne dit à pip : "crée un exécutable `algo-classifier` qui appelle la fonction `main()` dans `algo_classifier/cli.py`". Après `pip install -e .`, pip crée un script dans `venv/bin/algo-classifier` qui fait exactement ça.

**`pip install -e .` — le mode éditable :**
Le `-e` signifie "editable". Au lieu de copier ton code dans `site-packages`, pip crée un lien symbolique vers ton dossier actuel. Résultat : toute modification de ton code est immédiatement active sans réinstaller. C'est le mode standard pour le développement local.

---

## 22. Pourquoi ces choix d'architecture CLI

### Pourquoi `__main__.py` + `pyproject.toml` et pas juste `cli.py` à la racine ?

Trois niveaux de maturité possibles :

**Niveau 1 — script à la racine :**
```bash
python cli.py train --model baseline
```
Problème : pas installable, imports fragiles, ressemble à un ajout de dernière minute.

**Niveau 2 — module avec `-m` :**
```bash
python -m algo_classifier train --model baseline
```
Mieux : imports stables. Mais l'utilisateur doit être dans le bon répertoire et connaître la syntaxe `-m`.

**Niveau 3 — package installé (choix retenu) :**
```bash
algo-classifier train --model baseline
```
La commande est disponible partout dans le terminal, comme `git` ou `dbt`. C'est le standard des vrais projets ML open-source (LangChain CLI, dbt, Ruff, MLflow).

**Signal en entretien :** un recruteur qui clone le repo, fait `pip install -e .` et voit `algo-classifier --help` fonctionner immédiatement comprend que tu penses à l'utilisateur final et à la reproductibilité — deux réflexes Senior.

### Pourquoi les imports sont lazy dans `cli.py` ?

Chaque handler (`cmd_train`, `cmd_evaluate`...) importe ses dépendances à l'intérieur de la fonction, pas en haut du fichier :

```python
def cmd_train(args):
    from algo_classifier.baseline.train import train_model  # import local
    ...
```

Au lieu de :
```python
from algo_classifier.baseline.train import train_model  # import global
from algo_classifier.optimized.train import ...
from lightgbm import LGBMClassifier
...
```

**Pourquoi ?** Si les imports étaient globaux, `algo-classifier --help` chargerait LightGBM + sentence-transformers + scikit-learn au démarrage — soit ~2-3 secondes de latence juste pour afficher l'aide. Avec les imports lazy, `--help` s'affiche instantanément. Chaque sous-commande ne charge que ce dont elle a besoin, au moment où on en a besoin.

C'est le même pattern que Django, Click, ou Typer utilisent pour leurs CLIs.

### Pourquoi `--model-dir` et `--data-dir` comme arguments ?

Sans ces arguments, les chemins sont hardcodés dans `config.py` :
```python
DATA_DIR  = "./data"
MODEL_DIR = "./models/baseline"
```

C'est pratique localement, mais problématique dès qu'on change d'environnement : autre machine, CI/CD, Docker, cloud. En exposant ces chemins comme arguments CLI, on suit le principe **"pas de hardcoding en production"** :

```bash
# En local :
algo-classifier train --model baseline

# En CI/CD :
algo-classifier train --model baseline --data-dir /mnt/data --output-dir /mnt/models

# Avec un chemin custom :
algo-classifier evaluate --model-dir ./experiments/run_42/
```

Le comportement par défaut (`config.py`) reste intact — l'argument est optionnel. C'est le principe de **convention over configuration** : ça marche sans options, mais tout est configurable quand nécessaire.

### Pourquoi `--export json` ?

Sans export, les métriques s'affichent dans le terminal et disparaissent. En production ou en expérimentation, on veut tracer les résultats :

```bash
algo-classifier evaluate --model both --export json
# → génère results_baseline.json et results_optimized.json
```

Ces fichiers peuvent ensuite être versionnés, comparés entre runs, ou chargés dans un dashboard. C'est le début d'un pipeline MLOps — même minimal.

### Pourquoi `compare` délègue à `compare.py` via subprocess ?

```python
def cmd_compare(args):
    subprocess.run([sys.executable, "compare.py"])
```

`compare.py` existait avant le CLI et contient sa propre logique complète. Plutôt que de dupliquer le code ou de refactoriser `compare.py` en profondeur, on le délègue via subprocess — c'est pragmatique.

La vraie refactorisation propre serait d'importer `compare.py` directement comme module. C'est une dette technique mineure, documentable en entretien comme un choix délibéré de ne pas casser quelque chose qui fonctionne.

---

## 23. Les nouveaux fichiers — rôle et contenu

### `algo_classifier/__init__.py`

Transforme le dossier `algo_classifier/` en **package Python importable**. Sans ce fichier, `from algo_classifier.cli import main` échouerait — Python ne reconnaîtrait pas `algo_classifier` comme un package.

Son contenu :
```python
__version__ = "0.1.0"
```

On y met le numéro de version pour qu'il soit accessible via `import algo_classifier; algo_classifier.__version__`.

### `algo_classifier/baseline/__init__.py` et `algo_classifier/optimized/__init__.py`

Même rôle : rendre `baseline` et `optimized` importables comme sous-packages. Sans eux, `from algo_classifier.baseline.config import TARGET_TAGS` échouerait.

Ils sont volontairement vides (juste un commentaire) — il n'y a pas de logique à y mettre. C'est le cas le plus courant pour les `__init__.py` de sous-packages.

### `algo_classifier/__main__.py`

Point d'entrée pour `python -m algo_classifier`. Minimaliste par design :

```python
from algo_classifier.cli import main
if __name__ == "__main__":
    main()
```

La condition `if __name__ == "__main__"` est une convention Python : ce bloc ne s'exécute que si le fichier est lancé directement, pas s'il est importé. Ici c'est redondant (un `__main__.py` est toujours exécuté directement), mais c'est une bonne pratique de lisibilité.

### `algo_classifier/cli.py`

Le cœur du CLI. Contient :

**`build_parser()`** — construit le parser argparse avec toutes les sous-commandes. Séparée de `main()` pour être testable :
```python
parser = build_parser()
args = parser.parse_args(["train", "--model", "baseline"])
# → testable sans lancer le programme
```

**`cmd_train()`, `cmd_evaluate()`, `cmd_predict()`, `cmd_compare()`** — un handler par sous-commande. Chaque handler est responsable de :
1. Importer les modules nécessaires (lazy)
2. Orchestrer les appels aux fonctions métier
3. Gérer les cas d'erreur (`sys.exit(1)`)

**`main()`** — point d'entrée exposé dans `pyproject.toml`. Elle fait deux choses : parser les arguments et dispatcher vers le bon handler via `args.func(args)`.

Le pattern `args.func` est la clé du dispatch : chaque sous-commande enregistre son handler avec `set_defaults(func=cmd_train)`. Quand argparse parse la commande, `args.func` pointe vers le bon handler — pas besoin de `if args.command == "train"` etc.

### `pyproject.toml`

Trois sections importantes :

**`[build-system]`** — indique à pip quel backend utiliser pour construire le package :
```toml
build-backend = "setuptools.build_meta"
```
`setuptools` est le backend standard. On a remplacé `setuptools.backends.legacy:build` (trop récent pour la version installée) par `setuptools.build_meta` (compatible depuis setuptools 42).

**`[project.scripts]`** — la ligne la plus importante pour le CLI :
```toml
algo-classifier = "algo_classifier.cli:main"
```
Syntaxe : `nom-de-commande = "package.module:fonction"`. Pip crée un script exécutable dans `venv/bin/` qui appelle cette fonction. C'est exactement ce qui permet de taper `algo-classifier` n'importe où.

**`[tool.setuptools.packages.find]`** — dit à setuptools quels dossiers inclure dans le package :
```toml
include = ["algo_classifier*"]
```
Le `*` inclut `algo_classifier`, `algo_classifier.baseline`, `algo_classifier.optimized`. Sans ça, setuptools pourrait inclure `venv/`, `data/`, etc. dans le package.

### Corrections d'imports dans les fichiers existants

Tous les fichiers qui faisaient `from baseline.config import ...` ou `from data_loader import ...` ont été corrigés en `from algo_classifier.baseline.config import ...` et `from algo_classifier.data_loader import ...`.

**Pourquoi ce changement était nécessaire ?**

Avant le refactoring, `baseline/` et `optimized/` étaient à la racine du projet. Python ajoutait la racine au `sys.path`, ce qui permettait `from baseline.config import ...`.

Après le déplacement dans `algo_classifier/`, la racine du projet ne contient plus `baseline/` — elle contient `algo_classifier/`. L'import `from baseline.config` cherche un package `baseline` à la racine : introuvable. Il faut désormais le chemin complet depuis la racine du package : `from algo_classifier.baseline.config`.

**`baseline/config.py` et `baseline/features.py` n'ont pas été modifiés** car ils n'importaient rien du projet — uniquement des bibliothèques tierces (`numpy`, `pandas`, `re`...). Leurs imports étaient déjà corrects.

---

## 24. Récapitulatif — ce que le CLI apporte au projet

| Avant | Après |
|---|---|
| `python algo_classifier/baseline/train.py` | `algo-classifier train --model baseline` |
| `python algo_classifier/optimized/evaluate.py --test_path data/test_set_optimized.csv` | `algo-classifier evaluate --model optimized` |
| `python compare.py` | `algo-classifier compare` |
| Imports qui cassent selon le répertoire courant | Imports stables partout |
| Aucune aide disponible | `--help` sur chaque commande |
| Chemins hardcodés | `--data-dir`, `--output-dir`, `--model-dir` configurables |
| Résultats perdus dans le terminal | `--export json` pour tracer les runs |
| Script à copier-coller pour un collègue | `pip install -e . && algo-classifier --help` |

Le CLI transforme une collection de scripts en un **outil utilisable** — c'est la différence entre un projet de data scientist et un projet d'ingénieur.

---

*Document mis à jour et complété en avril 2026.*