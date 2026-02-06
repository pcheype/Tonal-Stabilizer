import cv2
import numpy as np
import math
from PySide6.QtCore import QThread, Signal

class TonalProcessor(QThread):
    progress_update = Signal(int)
    finished_signal = Signal(bool, str)

    def __init__(self, video_orig, work_w=120, sigma=0.10, omega_frac=0.7, lambda0=0.9):
        super().__init__()
        self.video_orig = video_orig
        # On ne passe plus video_stab ici, on le fera à l'enregistrement
        self.WORK_W = work_w
        self.SIGMA = sigma
        self.OMEGA_FRAC = omega_frac
        self.LAMBDA0 = lambda0
        
        # Stockage temporaire en RAM
        self.processed_frames = []
        self.fps = 30
        self.size = (0, 0)

    def run(self):
        """Lance le traitement uniquement (stockage en RAM)"""
        try:
            self.processed_frames = [] # Reset
            self.stabilize_to_memory()
            self.finished_signal.emit(True, "Traitement terminé en mémoire")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

    # --- Méthodes Statiques (inchangées) ---
    @staticmethod
    def resize_keep_aspect(bgr, w):
        h, W = bgr.shape[:2]
        if W == w: return bgr
        new_h = int(round(h * (w / float(W))))
        return cv2.resize(bgr, (w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def to_float01(bgr):
        return bgr.astype(np.float32) / 255.0

    @staticmethod
    def estimate_affine_dominant_motion(prev_gray, curr_gray):
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=800, qualityLevel=0.01, minDistance=8, blockSize=7)
        if p0 is None: return np.eye(3, dtype=np.float32)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
        if p1 is None or st is None: return np.eye(3, dtype=np.float32)
        st = st.reshape(-1).astype(bool)
        p0, p1 = p0[st].reshape(-1, 2), p1[st].reshape(-1, 2)
        if len(p0) < 12: return np.eye(3, dtype=np.float32)
        A, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        M = np.eye(3, dtype=np.float32)
        if A is not None: M[:2, :] = A.astype(np.float32)
        return M

    def build_overlap_correspondences(self, u_k, u_t, A_tk):
        Hk, Wk = u_k.shape[:2]
        Ht, Wt = u_t.shape[:2]
        xs, ys = np.meshgrid(np.arange(Wk, dtype=np.float32), np.arange(Hk, dtype=np.float32))
        ones = np.ones_like(xs)
        pts = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3).T
        q = (A_tk @ pts)
        qx, qy = q[0, :] / q[2, :], q[1, :] / q[2, :]
        inside = (qx >= 0) & (qx <= (Wt - 1)) & (qy >= 0) & (qy <= (Ht - 1))
        if not np.any(inside): return np.array([], dtype=np.int64), np.zeros((0, 2), dtype=np.int32), np.array([], dtype=bool)
        idx_x = np.where(inside)[0].astype(np.int64)
        qx_i, qy_i = np.rint(qx[inside]).astype(np.int32), np.rint(qy[inside]).astype(np.int32)
        y_int = np.stack([np.clip(qx_i, 0, Wt-1), np.clip(qy_i, 0, Ht-1)], axis=1)
        mu_k, mu_t = u_k.reshape(-1, 3).mean(axis=0), u_t.reshape(-1, 3).mean(axis=0)
        uk_flat = u_k.reshape(-1, 3)[idx_x]
        ut_flat = u_t.reshape(-1, 3)[(y_int[:, 1] * Wt + y_int[:, 0]).astype(np.int64)]
        good_mask = np.mean(((uk_flat - mu_k) - (ut_flat - mu_t)) ** 2, axis=1) < (self.SIGMA ** 2)
        return idx_x, y_int, good_mask

    @staticmethod
    def estimate_alpha_gamma_logLS(uk_vals, ut_vals, eps=1e-6):
        x, y = np.log(np.clip(ut_vals, eps, 1.0)), np.log(np.clip(uk_vals, eps, 1.0))
        x_m, y_m = x.mean(), y.mean()
        vx = ((x - x_m) ** 2).mean()
        gamma = ((x - x_m) * (y - y_m)).mean() / vx if vx > 1e-12 else 1.0
        alpha = np.exp(y_m - gamma * x_m)
        return float(alpha), float(gamma)

    @staticmethod
    def apply_parametric_correction_u8(frame_bgr_u8, alpha, gamma, lam):
        out = frame_bgr_u8.copy()
        x = np.arange(256, dtype=np.float32) / 255.0
        for c in range(3):
            y = np.clip(lam * (alpha[c] * (x ** gamma[c])) + (1.0 - lam) * x, 0.0, 1.0)
            out[:, :, c] = cv2.LUT(out[:, :, c], (y * 255.0 + 0.5).astype(np.uint8))
        return out

    @staticmethod
    def dominant_motion_vector(A_tk, W, H):
        cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
        q = A_tk @ np.array([cx, cy, 1.0], dtype=np.float32)
        return np.array([(q[0]/q[2]) - cx, (q[1]/q[2]) - cy], dtype=np.float32)

    # ----------------------------------------------------
    # PARTIE 1 : Traitement et stockage en RAM
    # ----------------------------------------------------
    def stabilize_to_memory(self):
        cap = cv2.VideoCapture(self.video_orig)
        if not cap.isOpened(): raise RuntimeError(f"Erreur ouverture: {self.video_orig}")

        self.fps = cap.get(cv2.CAP_PROP_FPS)
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (W, H)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ok, frame0 = cap.read()
        if not ok: return

        t, out_prev = 1, frame0.copy()
        self.processed_frames.append(out_prev) # Ajout à la liste
        
        u_k_lr = self.resize_keep_aspect(out_prev, self.WORK_W)
        prev_lr = u_k_lr.copy()
        A_tk = np.eye(3, dtype=np.float32)
        p_norm = float(max(u_k_lr.shape))

        while True:
            ok, frame_t = cap.read()
            if not ok: break
            t += 1

            if nframes > 0:
                self.progress_update.emit(int((t / nframes) * 100))

            curr_lr = self.resize_keep_aspect(frame_t, self.WORK_W)
            A_tt1 = self.estimate_affine_dominant_motion(cv2.cvtColor(prev_lr, cv2.COLOR_BGR2GRAY), 
                                                         cv2.cvtColor(curr_lr, cv2.COLOR_BGR2GRAY))
            A_tk = (A_tt1 @ A_tk).astype(np.float32)

            u_k_lr_f, u_t_lr_f = self.to_float01(u_k_lr), self.to_float01(curr_lr)
            idx_x, y_int, good = self.build_overlap_correspondences(u_k_lr_f, u_t_lr_f, A_tk)

            if len(idx_x) > 0 and np.count_nonzero(good) >= int(math.ceil(self.OMEGA_FRAC * len(idx_x))):
                idx_g = np.where(good)[0]
                uk_flat = u_k_lr_f.reshape(-1, 3)[idx_x[idx_g]]
                ut_flat = u_t_lr_f.reshape(-1, 3)[(y_int[idx_g, 1] * u_t_lr_f.shape[1] + y_int[idx_g, 0]).astype(np.int64)]
                
                V = self.dominant_motion_vector(A_tk, u_t_lr_f.shape[1], u_t_lr_f.shape[0])
                lam = np.clip(self.LAMBDA0 * math.exp(-np.linalg.norm(V) / max(1e-6, p_norm)), 0.0, 1.0)
                
                alpha, gamma = np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
                for c in range(3):
                    alpha[c], gamma[c] = self.estimate_alpha_gamma_logLS(uk_flat[:, c], ut_flat[:, c])

                out_t = self.apply_parametric_correction_u8(frame_t, alpha, gamma, lam)
                prev_lr = curr_lr
            else:
                out_t = frame_t.copy()
                u_k_lr = self.resize_keep_aspect(out_t, self.WORK_W)
                A_tk = np.eye(3, dtype=np.float32)
                prev_lr = u_k_lr.copy()

            self.processed_frames.append(out_t) # On stocke au lieu d'écrire sur disque

        cap.release()

    # ----------------------------------------------------
    # PARTIE 2 : Sauvegarde (Appelée par le bouton Télécharger)
    # ----------------------------------------------------
    def save_video(self, output_path):
        """Écrit les images stockées en RAM vers un fichier MP4"""
        if not self.processed_frames:
            return False, "Aucune donnée à enregistrer."
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.size)
            
            for frame in self.processed_frames:
                writer.write(frame)
                
            writer.release()
            return True, output_path
        except Exception as e:
            return False, str(e)