import cv2
import numpy as np
import math

class TonalProcessor:
    """
    Implémentation Alg.1 du papier FRIGO 2015
    Stabilisation tonale via mouvement dominant + correction paramétrique
    """

    def __init__(self, video_orig, video_stab,
                 work_w=120, sigma=0.10, omega_frac=0.7, lambda0=0.9):
        self.video_orig = video_orig
        self.video_stab = video_stab

        # Paramètres du papier
        self.WORK_W = work_w
        self.SIGMA = sigma
        self.OMEGA_FRAC = omega_frac
        self.LAMBDA0 = lambda0

    # ------------------------
    # Utils statiques
    # ------------------------
    @staticmethod
    def resize_keep_aspect(bgr, w):
        h, W = bgr.shape[:2]
        if W == w:
            return bgr
        new_h = int(round(h * (w / float(W))))
        return cv2.resize(bgr, (w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def to_float01(bgr):
        return bgr.astype(np.float32) / 255.0

    @staticmethod
    def estimate_affine_dominant_motion(prev_gray, curr_gray):
        p0 = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=800, qualityLevel=0.01, minDistance=8, blockSize=7
        )
        if p0 is None:
            return np.eye(3, dtype=np.float32)

        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
        if p1 is None or st is None:
            return np.eye(3, dtype=np.float32)

        st = st.reshape(-1).astype(bool)
        p0 = p0[st].reshape(-1, 2)
        p1 = p1[st].reshape(-1, 2)

        if len(p0) < 12:
            return np.eye(3, dtype=np.float32)

        A, _ = cv2.estimateAffinePartial2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0,
            maxIters=2000, confidence=0.99, refineIters=10
        )
        M = np.eye(3, dtype=np.float32)
        if A is not None:
            M[:2, :] = A.astype(np.float32)
        return M

    def build_overlap_correspondences(self, u_k, u_t, A_tk):
        Hk, Wk = u_k.shape[:2]
        Ht, Wt = u_t.shape[:2]

        xs, ys = np.meshgrid(np.arange(Wk, dtype=np.float32),
                             np.arange(Hk, dtype=np.float32))
        ones = np.ones_like(xs)
        pts = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3).T

        q = (A_tk @ pts)
        qx = q[0, :] / q[2, :]
        qy = q[1, :] / q[2, :]

        inside = (qx >= 0) & (qx <= (Wt - 1)) & (qy >= 0) & (qy <= (Ht - 1))
        if not np.any(inside):
            return np.array([], dtype=np.int64), np.zeros((0, 2), dtype=np.int32), np.array([], dtype=bool)

        idx = np.where(inside)[0]
        idx_x = idx.astype(np.int64)

        qx_i = np.rint(qx[inside]).astype(np.int32)
        qy_i = np.rint(qy[inside]).astype(np.int32)
        qx_i = np.clip(qx_i, 0, Wt - 1)
        qy_i = np.clip(qy_i, 0, Ht - 1)
        y_int = np.stack([qx_i, qy_i], axis=1)

        mu_k = u_k.reshape(-1, 3).mean(axis=0)
        mu_t = u_t.reshape(-1, 3).mean(axis=0)
        uk_flat = u_k.reshape(-1, 3)[idx_x]
        yt_lin = (y_int[:, 1] * Wt + y_int[:, 0]).astype(np.int64)
        ut_flat = u_t.reshape(-1, 3)[yt_lin]

        dk = uk_flat - mu_k
        dt = ut_flat - mu_t
        dist2 = np.mean((dk - dt) ** 2, axis=1)
        good_mask = dist2 < (self.SIGMA ** 2)

        return idx_x, y_int, good_mask

    @staticmethod
    def estimate_alpha_gamma_logLS(uk_vals, ut_vals, eps=1e-6):
        uk = np.clip(uk_vals, eps, 1.0)
        ut = np.clip(ut_vals, eps, 1.0)
        x = np.log(ut)
        y = np.log(uk)
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        vx = float(((x - x_mean) ** 2).mean())
        if vx < 1e-12:
            gamma = 1.0
            alpha = float(np.exp(y_mean - gamma * x_mean))
            return alpha, gamma
        cov = float(((x - x_mean) * (y - y_mean)).mean())
        gamma = cov / vx
        alpha = float(np.exp(y_mean - gamma * x_mean))
        return alpha, gamma

    @staticmethod
    def apply_parametric_correction_u8(frame_bgr_u8, alpha, gamma, lam):
        out = frame_bgr_u8.copy()
        x = np.arange(256, dtype=np.float32) / 255.0
        for c in range(3):
            y = lam * (alpha[c] * (x ** gamma[c])) + (1.0 - lam) * x
            y = np.clip(y, 0.0, 1.0)
            lut = (y * 255.0 + 0.5).astype(np.uint8)
            out[:, :, c] = cv2.LUT(out[:, :, c], lut)
        return out

    @staticmethod
    def dominant_motion_vector(A_tk, W, H):
        cx = (W - 1) * 0.5
        cy = (H - 1) * 0.5
        p = np.array([cx, cy, 1.0], dtype=np.float32)
        q = A_tk @ p
        qx = q[0] / q[2]
        qy = q[1] / q[2]
        return np.array([qx - cx, qy - cy], dtype=np.float32)

    # ------------------------
    # Algorithme principal
    # ------------------------
    def stabilize(self):
        cap = cv2.VideoCapture(self.video_orig)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir {self.video_orig}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.video_stab, fourcc, fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"Impossible d'écrire {self.video_stab}")

        # read first frame
        ok, frame0 = cap.read()
        if not ok:
            raise RuntimeError("Vidéo vide ?")

        t = 1
        k = 1
        out_prev = frame0.copy()
        writer.write(out_prev)
        u_k_full = out_prev.copy()
        u_k_lr = self.resize_keep_aspect(u_k_full, self.WORK_W)
        prev_lr = self.resize_keep_aspect(frame0, self.WORK_W)
        A_tk = np.eye(3, dtype=np.float32)
        p_norm = float(max(u_k_lr.shape[1], u_k_lr.shape[0]))

        while True:
            ok, frame_t = cap.read()
            if not ok:
                break
            t += 1

            curr_lr = self.resize_keep_aspect(frame_t, self.WORK_W)
            prev_gray = cv2.cvtColor(prev_lr, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_lr, cv2.COLOR_BGR2GRAY)
            A_tt1 = self.estimate_affine_dominant_motion(prev_gray, curr_gray)
            A_tk = (A_tt1 @ A_tk).astype(np.float32)

            u_k_lr_f = self.to_float01(u_k_lr)
            u_t_lr_f = self.to_float01(curr_lr)
            idx_x, y_int, good = self.build_overlap_correspondences(u_k_lr_f, u_t_lr_f, A_tk)

            Omega_size = int(len(idx_x))
            Omega_tk_size = int(np.count_nonzero(good))

            if Omega_size > 0 and Omega_tk_size >= int(math.ceil(self.OMEGA_FRAC * Omega_size)):
                idx_good = np.where(good)[0]
                idx_x_good = idx_x[idx_good]
                Ht_lr, Wt_lr = u_t_lr_f.shape[:2]
                yt = y_int[idx_good]
                yt_lin = (yt[:, 1] * Wt_lr + yt[:, 0]).astype(np.int64)
                uk_flat = u_k_lr_f.reshape(-1, 3)[idx_x_good]
                ut_flat = u_t_lr_f.reshape(-1, 3)[yt_lin]

                V = self.dominant_motion_vector(A_tk, Wt_lr, Ht_lr)
                vnorm = float(np.linalg.norm(V))
                lam = float(self.LAMBDA0 * math.exp(-vnorm / max(1e-6, p_norm)))
                lam = max(0.0, min(1.0, lam))

                alpha = np.zeros(3, dtype=np.float32)
                gamma = np.zeros(3, dtype=np.float32)
                for c in range(3):
                    a, g = self.estimate_alpha_gamma_logLS(uk_flat[:, c], ut_flat[:, c])
                    alpha[c] = a
                    gamma[c] = g

                out_t = self.apply_parametric_correction_u8(frame_t, alpha, gamma, lam)
                writer.write(out_t)
                out_prev = out_t
                prev_lr = curr_lr

            else:
                # même logique que le script original pour changer la référence
                k = t - 1
                u_k_full = out_prev.copy()
                u_k_lr = self.resize_keep_aspect(u_k_full, self.WORK_W)
                A_tk = np.eye(3, dtype=np.float32)
                A_tk = A_tt1.copy()
                u_k_lr_f = self.to_float01(u_k_lr)
                u_t_lr_f = self.to_float01(curr_lr)
                idx_x, y_int, good = self.build_overlap_correspondences(u_k_lr_f, u_t_lr_f, A_tk)
                Omega_size = int(len(idx_x))
                Omega_tk_size = int(np.count_nonzero(good))
                if Omega_size > 0 and Omega_tk_size >= int(math.ceil(self.OMEGA_FRAC * Omega_size)):
                    idx_good = np.where(good)[0]
                    idx_x_good = idx_x[idx_good]
                    Ht_lr, Wt_lr = u_t_lr_f.shape[:2]
                    yt = y_int[idx_good]
                    yt_lin = (yt[:, 1] * Wt_lr + yt[:, 0]).astype(np.int64)
                    uk_flat = u_k_lr_f.reshape(-1, 3)[idx_x_good]
                    ut_flat = u_t_lr_f.reshape(-1, 3)[yt_lin]
                    V = self.dominant_motion_vector(A_tk, Wt_lr, Ht_lr)
                    vnorm = float(np.linalg.norm(V))
                    lam = float(self.LAMBDA0 * math.exp(-vnorm / max(1e-6, p_norm)))
                    lam = max(0.0, min(1.0, lam))
                    alpha = np.zeros(3, dtype=np.float32)
                    gamma = np.zeros(3, dtype=np.float32)
                    for c in range(3):
                        a, g = self.estimate_alpha_gamma_logLS(uk_flat[:, c], ut_flat[:, c])
                        alpha[c] = a
                        gamma[c] = g
                    out_t = self.apply_parametric_correction_u8(frame_t, alpha, gamma, lam)
                else:
                    out_t = frame_t.copy()

                writer.write(out_t)
                out_prev = out_t
                prev_lr = curr_lr
                u_k_full = out_prev.copy()
                u_k_lr = self.resize_keep_aspect(u_k_full, self.WORK_W)
                A_tk = np.eye(3, dtype=np.float32)

        cap.release()
        writer.release()
        print("OK ->", self.video_stab)


if __name__ == "__main__":
    VIDEO_ORIG = r"C:\Users\pchey\Desktop\TIVO\Tonal Stabilizer\stabilizer\out.mp4"
    VIDEO_STAB = r"C:\Users\pchey\Desktop\TIVO\Tonal Stabilizer\stabilizer\outstab.mp4"
    processor = TonalProcessor(VIDEO_ORIG, VIDEO_STAB)
    processor.stabilize()
