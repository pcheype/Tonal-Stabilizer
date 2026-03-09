# Tonal Stabilizer

Application GUI pour la stabilisation tonale de vidéos et de séquences d’images.

## Description

Tonal Stabilizer est un outil conçu pour réduire les fluctuations tonales dans les vidéos.  
Il propose une interface graphique intuitive basée sur **PySide6** et utilise **OpenCV** et **NumPy** pour le traitement d’images.

## Fonctionnalités

- **Importation vidéo** : support des fichiers MP4 et des dossiers d’images  
- **Traitement tonal** : stabilisation paramétrable avec :
  - λ (lambda) – facteur de régularisation  
  - ω (omega) – paramètre de recouvrement  
  - σ (sigma)
- **Prévisualisation** : affichage des vidéos originales et traitées  
- **Export de comparaison** : génération de vidéos de comparaison diagonale  
- **Exportation** : sauvegarde des vidéos stabilisées au format MP4

---

# Installation

Ce projet utilise **uv** pour la gestion des dépendances.

### 1. Cloner le dépôt

```bash
git clone https://github.com/pcheype/Tonal-Stabilizer.git
cd Tonal-Stabilizer
```

### 2. Lancer l’application

```bash
uv run python -m stabilizer
```

`uv` créera automatiquement l’environnement et installera toutes les dépendances nécessaires.

---

# Utilisation

## 1 — Importation

- Charger une **vidéo MP4** ou un **dossier d’images**  
- Prévisualiser la séquence d’entrée

## 2 — Traitement

Ajuster les paramètres de stabilisation :

- **λ (Lambda)** — facteur de régularisation (défaut : 0.9)  
- **ω (Omega)** — paramètre de recouvrement (défaut : 0.7)  
- **σ (Sigma)** — paramètre de lissage (défaut : 0.1)  

Puis cliquer sur **Start Processing**.

## 3 — Exportation

- Exporter la vidéo stabilisée  
- Générer éventuellement une **vidéo de comparaison**

---

# Structure du projet

```
Tonal-Stabilizer
│
├── pyproject.toml
├── README.md
│
├── stabilizer
│   ├── __init__.py
│   ├── mainwindow.py
│   ├── frameloader.py
│   ├── tonalprocessor.py
│   └── formator.py
│
└── data
```

---

# Dépendances principales

- **PySide6** — framework GUI  
- **OpenCV** — traitement vidéo et images  
- **NumPy** — calculs numériques  

---

# Auteurs

Paolo Cheype  
Louis Dorlencourt

---
