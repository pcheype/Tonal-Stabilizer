:::mermaid
graph TD
%% Configuration du style de texte
accTitle: Architecture Tonal Stabilizer
accDescr: Diagramme d'architecture avec texte noir pour une meilleure lisibilité.

subgraph Data_Storage [📂 Dossier Data]
    D1[Images_graycard]
    D2[Images_planthouse]
    D3[Images_weirdhouse]
end

subgraph App_Execution [🚀 Tonal Stabilizer]
    Main[__main__.py<br/><i>Point d'entrée</i>]
    UI[mainwindow.py<br/><i>Interface utilisateur</i>]
end

subgraph Processing_Pipeline [⚙️ Pipeline de Traitement]
    Loader[frameloader.py<br/><b>1. Lecture Séquences</b>]
    Proc[tonalprocessor.py<br/><b>2. Traitement d'images</b>]
    Form[formator.py<br/><b>3. Sortie & Export</b>]
end

%% Flux de données
Main --> UI
UI --> Loader
D1 & D2 & D3 -.-> Loader
Loader --> Proc
Proc --> Form
Form --> UI

%% Application du texte NOIR sur tous les éléments
classDef blackText fill:#fff,stroke:#333,stroke-width:2px,color:#000;
class D1,D2,D3,Main,UI,Loader,Proc,Form blackText;

%% Surcharge de couleurs pour les blocs spécifiques (en gardant le texte noir)
style UI fill:#e1f5fe,color:#000
style Proc fill:#fff3e0,color:#000
style Loader fill:#e8f5e9,color:#000
style Form fill:#fce4ec,color:#000
:::