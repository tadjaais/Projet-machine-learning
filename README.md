📘 Prévision des Ventes Fibre – Comparaison des Modèles de Séries Temporelles et de Machine Learning

Projet de Data Science appliqué au secteur des télécommunications en Côte d’Ivoire

🎯 Contexte et Objectif

Le secteur des télécommunications en Côte d’Ivoire connaît une transformation majeure, portée par la digitalisation et l’adoption croissante de la Fibre optique et de la 4G. Dans ce contexte de forte concurrence, la capacité à anticiper les ventes constitue un levier stratégique essentiel pour :
-optimiser la planification,
-ajuster les objectifs commerciaux,
-mieux allouer les ressources,
-orienter les actions marketing,
-et améliorer la satisfaction client.

Ce projet vise à comparer deux grandes familles de modèles de prévision appliquées aux ventes Fibre d’Orange CI :

-Les modèles de séries temporelles (ARIMA, SARIMA, Holt-Winters)

-Les modèles de Machine Learning (Régression Linéaire, Random Forest, XGBoost)

L’objectif est d’identifier laquelle de ces approches fournit les prévisions les plus fiables, les plus pertinentes et les plus adaptées au contexte télécom ivoirien.

🧩 Problématique

Quelle approche – modèle de série temporelle ou Machine Learning – permet de fournir les prévisions les plus fiables et opérationnellement utiles pour les ventes Fibre chez Orange CI ?

🎯 Objectifs spécifiques

Construire et entraîner plusieurs modèles de prévision issus des deux familles (Time Series & Machine Learning).

Comparer leurs performances à l’aide de métriques standardisées : MAE, RMSE, MAPE.

Analyser leurs forces, limites et pertinence.

Identifier l’approche la plus adaptée à la dynamique réelle des ventes Fibre en Côte d’Ivoire.

Fournir un cadre méthodologique reproductible et exploitable par les équipes commerciales et BI.

🛠️ Méthodologie
1️⃣ Analyse exploratoire

<-Inspection des tendances et saisonnalités

<-Détection des ruptures et pics commerciaux

<-Analyse des corrélations avec les facteurs commerciaux

<-Visualisation des cycles mensuels

2️⃣ Prétraitement

-Nettoyage des données

-Transformation des dates

-Création de variables dérivées (mois, trimestre, campagnes marketing, zones géographiques, etc.)

-Normalisation / scalage pour les modèles ML

3️⃣ Modélisation
✔️ Modèles de séries temporelles

Holt-Winters (additif/multiplicatif)

ARIMA / SARIMA

Analyse des résidus et des diagnostics

✔️ Modèles Machine Learning

Régression Linéaire

Random Forest Regressor

XGBoost Regressor

4️⃣ Évaluation

Métriques utilisées :

MAE (Mean Absolute Error)

RMSE (Root Mean Square Error)

MAPE (Mean Absolute Percentage Error)

Visualisations :

Observé vs Prédit

Courbes d’erreur

Importance des variables (ML)

🏆 Résultats clés

Les résultats montrent que, malgré la nature temporelle des ventes :

⭐ La Régression Linéaire (Machine Learning) obtient les meilleures performances.

Cela s’explique par :

le poids important des facteurs commerciaux, les influences marketing (promotions, campagnes),les disparités entre zones géographiques,
des comportements clients non linéaires, difficiles à capturer pour les modèles entièrement temporels.

Les modèles ARIMA/SARIMA restent performants mais présentent des limitations lorsque des variables explicatives externes jouent un rôle majeur.


📂 Structure du repository

.
├── notebooks/
│   └── Prevision_Ventes_Fibre.ipynb      # Notebook principal (code + sorties)
├── data/
│   └── ventes_fibre.csv                  # Données (si légères / anonymisées)
├── src/
│   ├── preprocessing.py                  # Pipeline de préparation
│   ├── timeseries_models.py              # ARIMA, SARIMA, Holt-Winters
│   └── ml_models.py                      # Linear Regression, Random Forest, XGBoost
├── results/
│   ├── figures/                          # Graphiques générés
│   └── predictions.csv                   # Sorties des modèles
├── requirements.txt                       # Bibliothèques nécessaires
└── README.md                              # Documentation


⚙️ Installation & Exécution

1️⃣ Cloner le projet
git clone https://github.com/ton-utilisateur/ton-repo.git
cd ton-repo

2️⃣ Installer les dépendances
pip install -r requirements.txt

3️⃣ Lancer le notebook
Ouvrir :
ML.ipynb

📦 Technologies utilisées

Python 3

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Statsmodels

XGBoost

Prophet

✍️ Auteure

Raoufa Touré
Réalisé dans le cadre d’un projet  Data Science, appliqué au secteur télécom ivoirien.

🚀 Perspectives

Ajout d’un modèle LSTM (Deep Learning)

Intégration d’un dashboard Streamlit interactif

Automatisation des prévisions via API ( mise en production )

Prise en compte de nouvelles variables marketing





