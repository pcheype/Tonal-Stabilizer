import cv2
import numpy as np
import os
from PySide6.QtCore import QThread, Signal

class FormatorProcessor(QThread):
    """
    Génère une comparaison diagonale entre deux vidéos.
    Supporte le threading et la séparation calcul/écriture pour PySide6.
    """
    progress_update = Signal(int)
    finished_signal = Signal(bool, str)

    def __init__(self, video_orig, video_stab, resize_to=None, diag="\\"):
        super().__init__()
        self.video_orig = video_orig
        self.video_stab = video_stab
        self.resize_to = resize_to
        self.diag = diag
        
        # Stockage pour le bouton "Télécharger"
        self.processed_frames = []
        self.fps = 30.0
        self.final_size = (0, 0)

    def run(self):
        """Exécution du calcul en arrière-plan"""
        try:
            self.processed_frames = [] # Reset RAM
            self.generate_composite_frames()
            self.finished_signal.emit(True, "Comparaison générée avec succès.")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

    # --- Logique de traitement (Adaptée de ta classe Formator) ---

    def generate_composite_frames(self):
        cap_o = cv2.VideoCapture(self.video_orig)
        cap_s = cv2.VideoCapture(self.video_stab)

        if not cap_o.isOpened() or not cap_s.isOpened():
            raise IOError("Impossible d'ouvrir l'une des deux vidéos.")

        # Récupération des métadonnées
        fps_o = cap_o.get(cv2.CAP_PROP_FPS)
        fps_s = cap_s.get(cv2.CAP_PROP_FPS)
        nframes = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.fps = min(fps_o, fps_s) if fps_o > 0 else 30.0
        frame_idx = 0

        while True:
            ret_o, fo = cap_o.read()
            ret_s, fs = cap_s.read()

            if not ret_o or not ret_s:
                break

            # 1. Mise à la même taille
            fo, fs = self.ensure_same_size(fo, fs, self.resize_to)
            
            # 2. Création du composite diagonal
            combined = self.diagonal_composite(fo, fs, diag=self.diag)
            
            # 3. Ajout des labels et du compteur
            combined = self.draw_labels(combined)
            cv2.putText(combined, f"Frame {frame_idx}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 4. Stockage en RAM
            self.processed_frames.append(combined)
            
            # 5. Progression UI
            frame_idx += 1
            if nframes > 0:
                self.progress_update.emit(int((frame_idx / nframes) * 100))

        if self.processed_frames:
            h, w = self.processed_frames[0].shape[:2]
            self.final_size = (w, h)

        cap_o.release()
        cap_s.release()

    # --- Méthodes de dessin et calcul ---

    def ensure_same_size(self, a, b, resize_to=None):
        if resize_to is not None:
            a = cv2.resize(a, resize_to, interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, resize_to, interpolation=cv2.INTER_AREA)
        ha, wa = a.shape[:2]
        hb, wb = b.shape[:2]
        h, w = min(ha, hb), min(wa, wb)
        return a[:h, :w], b[:h, :w]

    def diagonal_composite(self, orig_bgr, stab_bgr, diag="\\", line_thickness=2):
        h, w = orig_bgr.shape[:2]
        yy, xx = np.indices((h, w), dtype=np.float32)
        m = (h - 1) / max((w - 1), 1)
        
        boundary = m * xx
        mask_orig = yy > boundary
        mask_stab = ~mask_orig
        
        out = np.empty_like(orig_bgr)
        out[mask_orig] = orig_bgr[mask_orig]
        out[mask_stab] = stab_bgr[mask_stab]
        
        # Ligne de séparation blanche
        dist = np.abs(yy - boundary)
        out[dist <= line_thickness] = (255, 255, 255)
        return out

    def draw_labels(self, frame, labels=("Original", "Stabilized")):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, w / 1200)
        color = (0, 255, 0)
        
        # Original (Bas-Gauche)
        cv2.putText(frame, labels[0], (15, h - 20), font, font_scale, color, 2, cv2.LINE_AA)
        
        # Stabilized (Haut-Droite)
        (tw, th), _ = cv2.getTextSize(labels[1], font, font_scale, 2)
        cv2.putText(frame, labels[1], (w - tw - 15, th + 20), font, font_scale, color, 2, cv2.LINE_AA)
        
        return frame

    # --- Sauvegarde Finale ---

    def save_comparison_video(self, output_path):
        """Action liée au bouton 'Télécharger'"""
        if not self.processed_frames:
            return False, "Aucune frame en mémoire."
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.final_size)
            
            for f in self.processed_frames:
                writer.write(f)
                
            writer.release()
            return True, output_path
        except Exception as e:
            return False, str(e)