# Tonal Stabilizer

Une application GUI pour stabilisation tonale de vidéos et d'images.

## Description

Tonal Stabilizer est un outil conçu pour améliorer la qualité des vidéos en stabilisant les variations tonales. L'application offre une interface utilisateur intuitive basée sur PySide6 et utilise OpenCV et NumPy pour les traitements vidéo.

### Fonctionnalités

- **Importation vidéo** : Supporte les fichiers MP4 et les dossiers d'images
- **Traitement tonal** : Stabilisation paramétrable avec contrôles de régularisation, recouvrement et sigma
- **Visualisation** : Prévisualisation des vidéos originales et traitées
- **Comparaison** : Génération de vidéos de comparaison diagonale
- **Exportation** : Sauvegarde des résultats au format MP4

## Installation

### 1. Prérequis

- Python 3.8 ou supérieur
- pip

### 2. Cloner ou télécharger le projet

```bash
cd "Chemin\vers\Tonal Stabilizer"
```

### 3. Créer et activer l'environnement virtuel

#### PowerShell (Windows)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

#### CMD (Windows)
```cmd
python -m venv .venv
.venv\Scripts\activate
```

#### Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Installer les dépendances

#### Option A : Installer depuis `pyproject.toml` (recommandé)
```bash
pip install --upgrade pip
pip install -e .
```

#### Option B : Installer depuis `requirements.txt`
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Utilisation

### Lancer l'application

```bash
stabilizer
```

Ou directement avec Python :
```bash
python -m stabilizer.mainwindow
```

### Étapes d'utilisation

1. **Étape 1 : Importation**
   - Sélectionnez une vidéo MP4 ou un dossier d'images
   - Prévisualisez votre vidéo

2. **Étape 2 : Processing**
   - Ajustez les paramètres :
     - **Λ (Lambda)** : Facteur de régularisation (défaut : 0.9)
     - **ω (Omega)** : Paramètre de recouvrement (défaut : 0.7)
     - **σ (Sigma)** : Sigma (défaut : 0.1)
   - Cliquez sur "Start Processing"

3. **Étape 3 : Exportation**
   - Téléchargez votre vidéo stabilisée
   - Générez une comparaison et téléchargez-la

## Structure du projet

```
Tonal Stabilizer/
├── pyproject.toml          # Configuration du projet
├── requirements.txt        # Dépendances Python
├── README.md              # Ce fichier
├── stabilizer/
│   ├── __init__.py
│   ├── mainwindow.py      # Interface utilisateur principale
│   ├── frameloader.py     # Gestion des vidéos et images
│   ├── tonalprocessor.py  # Moteur de traitement tonal
│   └── formator.py        # Génération de comparaisons
├── test_stabilizer/       # Tests (à développer)
└── data/
    ├── Images_graycard/   # Images de test
    └── Images_weirdhouse/ # Images de test
```

## Développement

### Installer en mode développeur

```bash
pip install -e ".[dev]"
```

Cela installe également les outils de développement : `pytest`, `black`, `flake8`.

### Exécuter les tests

```bash
pytest
```

### Formater le code

```bash
black stabilizer/
```

### Vérifier la qualité du code

```bash
flake8 stabilizer/
```

## Dépendances principales

- **PySide6** : Framework GUI
- **OpenCV** : Traitement vidéo et images
- **NumPy** : Calculs numériques

## Auteurs

Paolo Cheype, Louis Dorlencourt

## Licence

MIT

## Notes

- Les vidéos sont traitées et stockées temporairement en mémoire RAM
- Assurez-vous d'avoir suffisamment d'espace mémoire pour les vidéos longues
- Les formats MP4 convertis utilisent le codec `mp4v` pour une meilleure compatibilité
