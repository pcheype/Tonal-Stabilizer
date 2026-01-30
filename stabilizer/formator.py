import cv2
import numpy as np
from tkinter import Tk, filedialog
import os


class Formator:

    def __init__(self):
        pass

    # -------------------------------
    # Ouvrir une vidéo
    # -------------------------------
    def open_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la vidéo: {video_path}")
        return cap

    # -------------------------------
    # Obtenir infos vidéo
    # -------------------------------
    def get_video_info(self, cap):
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, w, h, n

    # -------------------------------
    # Assurer même taille pour 2 frames
    # -------------------------------
    def ensure_same_size(self, a, b, resize_to=None):
        if resize_to is not None:
            a = cv2.resize(a, resize_to, interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, resize_to, interpolation=cv2.INTER_AREA)
        ha, wa = a.shape[:2]
        hb, wb = b.shape[:2]
        h = min(ha, hb)
        w = min(wa, wb)
        return a[:h, :w], b[:h, :w]

    # -------------------------------
    # Création du composite diagonal
    # -------------------------------
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
        dist = np.abs(yy - boundary)
        line_mask = dist <= line_thickness
        out[line_mask] = (255, 255, 255)
        return out

    # -------------------------------
    # Ajouter labels sur la frame
    # -------------------------------
    def draw_labels(self, frame, labels=("Original", "Stabilized")):
        h, w = frame.shape[:2]
        
        color = (0, 255, 0)  # vert
        thickness = 2
        font_scale = max(0.5, w / 1000)  # adapte la taille selon la largeur

        # Original : bas à gauche
        text_orig = labels[0]
        (text_w, text_h), _ = cv2.getTextSize(text_orig, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x_orig = 10  # 10 pixels du bord gauche
        y_orig = h - 10  # 10 pixels du bord bas
        cv2.putText(frame, text_orig, (x_orig, y_orig), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

        # Stabilized : haut à droite
        text_stab = labels[1]
        (text_w2, text_h2), _ = cv2.getTextSize(text_stab, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x_stab = w - text_w2 - 10  # 10 pixels du bord droit
        y_stab = 10 + text_h2      # 10 pixels du bord haut
        cv2.putText(frame, text_stab, (x_stab, y_stab), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)

        return frame



    # -------------------------------
    # Comparer deux vidéos
    # -------------------------------
    def compare_videos_diagonal(self, video_orig, video_stab,
                                resize_to=None,
                                diag="\\",
                                output_path="comparison.mp4",
                                window_name="Diagonal Comparison"):

        cap_o = self.open_video(video_orig)
        cap_s = self.open_video(video_stab)

        fps_o, wo, ho, no = self.get_video_info(cap_o)
        fps_s, ws, hs, ns = self.get_video_info(cap_s)
        fps = min(fps_o, fps_s) if fps_o > 0 and fps_s > 0 else 30.0
        delay = max(1, int(1000 / fps))

        print(f"[INFO] Orig: {os.path.basename(video_orig)} {wo}x{ho} FPS={fps_o:.1f}")
        print(f"[INFO] Stab: {os.path.basename(video_stab)} {ws}x{hs} FPS={fps_s:.1f}")
        print(f"[INFO] FPS lecture: {fps}")
        print(f"[INFO] Diagonale: {diag}")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        paused = False
        frame_idx = 0
        out = None

        while True:
            if not paused:
                ret_o, fo = cap_o.read()
                ret_s, fs = cap_s.read()
                if not ret_o or not ret_s:
                    print("[INFO] Fin d'une des vidéos.")
                    break

                fo, fs = self.ensure_same_size(fo, fs, resize_to)
                combined = self.diagonal_composite(fo, fs, diag=diag)
                combined = self.draw_labels(combined)

                cv2.putText(combined, f"Frame {frame_idx}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if out is None:
                    h, w = combined.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                    if not out.isOpened():
                        raise IOError(f"Impossible d'ouvrir le writer: {output_path}")
                    print(f"[INFO] Enregistrement vidéo -> {output_path} ({w}x{h} @ {fps:.2f} fps)")

                cv2.imshow(window_name, combined)
                out.write(combined)
                frame_idx += 1

            key = cv2.waitKey(delay) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                cap_o.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap_s.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
            elif key == ord('a'):
                pos = max(frame_idx - 2, 0)
                cap_o.set(cv2.CAP_PROP_POS_FRAMES, pos)
                cap_s.set(cv2.CAP_PROP_POS_FRAMES, pos)
                frame_idx = pos

        cap_o.release()
        cap_s.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print("[OK] Comparaison terminée !")

if __name__ == "__main__":

    formatter = Formator()
    video_orig = r"C:\Users\pchey\Desktop\TIVO\Tonal Stabilizer\stabilizer\out.mp4"
    video_stab = r"C:\Users\pchey\Desktop\TIVO\Tonal Stabilizer\stabilizer\outstab.mp4"
    output = r"C:\Users\pchey\Desktop\TIVO\Tonal Stabilizer\stabilizer\comparison.mp4"

    formatter.compare_videos_diagonal(video_orig, video_stab, output_path=output)

