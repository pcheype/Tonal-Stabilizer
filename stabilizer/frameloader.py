import cv2
import os
from pathlib import Path
from PySide6.QtWidgets import QFileDialog, QMessageBox

class VideoManager:
    
    def __init__(self, parent=None):
        self.parent = parent # Référence à la MainWindow si nécessaire
        self.supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tif")

    def get_path_from_dialog(self, mode="folder"):
        """Centralise les demandes de sélection de fichiers/dossiers via PySide6."""
        if mode == "folder":
            return QFileDialog.getExistingDirectory(self.parent, "Sélectionner le dossier d'images")
        elif mode == "video":
            path, _ = QFileDialog.getOpenFileName(self.parent, "Sélectionner une vidéo", "", "Videos (*.mp4 *.avi *.mov)")
            return path
        elif mode == "save":
            path, _ = QFileDialog.getSaveFileName(self.parent, "Enregistrer la vidéo", "output.mp4", "Video (*.mp4)")
            return path
        return None

    def images_to_video(self, input_folder, output_path, fps=30):
        """Convertit un dossier d'images en vidéo MP4 de manière optimisée."""
        folder_path = Path(input_folder)
        # Tri naturel des fichiers pour éviter l'ordre 1, 10, 2
        files = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in self.supported_formats])

        if not files:
            return False, "Aucune image trouvée dans le dossier."

        # Lecture de la première image pour les dimensions
        first_frame = cv2.imread(str(files[0]))
        if first_frame is None:
            return False, "Erreur lors de la lecture de la première image."
            
        h, w = first_frame.shape[:2]
        
        # Initialisation du writer (FourCC mp4v est très compatible)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        try:
            for f in files:
                img = cv2.imread(str(f))
                if img is not None:
                    # On s'assure que toutes les images ont la même taille
                    if (img.shape[1], img.shape[0]) != (w, h):
                        img = cv2.resize(img, (w, h))
                    writer.write(img)
            return True, output_path
        except Exception as e:
            return False, str(e)
        finally:
            writer.release()

    def process_existing_video(self, input_path, output_path):
        """Permet de traiter ou simplement copier/renommer une vidéo existante."""
        # Ici on peut imaginer un simple transcodage ou une copie
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        
        cap.release()
        writer.release()
        return True, output_path