# face_solver.py
# ==============
# Tier-3 face refinement: fit ICT-FaceKit expression weights to the
# stabilized MediaPipe landmarks, per frame.
#
# The baked basis (pipeline/ict_face_basis.npz, built dev-side by
# bake_ict_basis.py from the MIT-licensed ICT-FaceKit) stores, for a
# subset of MediaPipe landmark indices, the ICT neutral position and
# each expression shape's displacement — all in the same canonical
# frame stabilize_landmarks() produces (eye-mid origin, interocular=1,
# +X subject-left, +Y up, +Z forward).
#
# Per frame we solve a box-constrained ridge least squares
#     min_w || W_r (D w - b) ||^2 + lam ||w||^2,   0 <= w <= 1
# where D is the (3M x K) delta basis, b the observed landmark offsets
# from the actor's own neutral template, and W_r per-landmark/axis
# weights. Because the basis deltas are motion vectors relative to each
# face's OWN neutral, the actor-vs-ICT identity difference largely
# cancels — no identity fit is needed.
#
# Solved with batched projected Gauss-Seidel on the K x K normal
# equations: the Gram matrix is built once, every frame updates
# simultaneously as (N,K) numpy ops, so whole-clip solves take seconds.
# No temporal term here — the export pipeline's Savitzky-Golay pass
# (smooth_face_data) already smooths the solved channels.

import json
import os

import numpy as np

from pipeline import face_landmark_topology as topo

BASIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "ict_face_basis.npz")

# Per-region row weights: where landmark evidence is trusted for the
# expression solve. Eye rings get LOW weight — blinks stay MediaPipe's
# job, and eyelid z-noise otherwise bleeds into brow/cheek channels.
_REGION_WEIGHTS = (
    (1.6, set(topo.LIPS_OUTER_RING) | set(topo.LIPS_INNER_RING)),
    (1.5, set(topo.CHIN_CLUSTER) | {175, 199, 200, 18, 32, 194,
                                    201, 208, 421, 418, 424, 262}),
    (1.2, set(topo.LEFT_BROW_LOWER) | set(topo.LEFT_BROW_UPPER)
     | set(topo.RIGHT_BROW_LOWER) | set(topo.RIGHT_BROW_UPPER)),
    (1.1, set(topo.NOSE_WING_LEFT) | set(topo.NOSE_WING_RIGHT)
     | {1, 2, 168, 6, 197, 195}),
    (0.35, set(topo.LEFT_EYE_RING) | set(topo.RIGHT_EYE_RING)),
)
_DEFAULT_REGION_WEIGHT = 0.65
_Z_AXIS_WEIGHT = 0.35          # MediaPipe z is the weakest signal


def load_basis(path=None):
    """Load the baked ICT basis. Returns dict or None (missing file)."""
    p = path or BASIS_PATH
    if not os.path.isfile(p):
        return None
    b = np.load(p, allow_pickle=True)
    basis = {
        "mp_indices": b["mp_indices"].astype(int),      # (M,)
        "neutral": b["neutral"].astype(np.float64),     # (M, 3)
        "deltas": b["deltas"].astype(np.float64),       # (K, M, 3)
        "names": [str(n) for n in b["names"]],
    }
    try:
        basis["meta"] = json.loads(str(b["meta"]))
    except Exception:
        basis["meta"] = {}
    return basis


def _row_weights(mp_indices):
    """Per-landmark weight vector (M,) from the region table."""
    w = np.full(len(mp_indices), _DEFAULT_REGION_WEIGHT, dtype=np.float64)
    for weight, region in _REGION_WEIGHTS:
        mask = np.array([int(i) in region for i in mp_indices])
        w[mask] = weight
    return w


def solve_expressions(stab, template, detected, basis,
                      ridge=0.03, sweeps=60, tol=1e-4,
                      progress_cb=None):
    """Fit expression weights for every detected frame.

    stab:      (N, 478, 3) canonical-frame landmarks (stabilize_landmarks)
    template:  (478, 3) actor's neutral template, same frame
    detected:  (N,) bool
    basis:     dict from load_basis()

    Returns (weights (N, K) float32 in [0,1], names list, diag dict).
    Undetected frames stay all-zero.
    """
    mp_idx = basis["mp_indices"]
    D = basis["deltas"]                                # (K, M, 3)
    names = basis["names"]
    K, M, _ = D.shape

    det = np.asarray(detected, dtype=bool)
    n = len(stab)
    det_rows = np.flatnonzero(det)

    # Row weights: per landmark x per axis.
    w_lm = _row_weights(mp_idx)                        # (M,)
    w_axis = np.array([1.0, 1.0, _Z_AXIS_WEIGHT])
    W = (w_lm[:, None] * w_axis[None, :]).reshape(-1)  # (3M,)

    A = D.reshape(K, -1).T * W[:, None]                # (3M, K)
    G = A.T @ A                                        # (K, K)
    lam = ridge * float(np.trace(G)) / K
    G_reg = G + lam * np.eye(K)
    g_diag = np.diag(G_reg).copy()

    # Observations: actor deltas from their own neutral, weighted.
    obs = (stab[det_rows][:, mp_idx, :]
           - template[mp_idx][None, :, :]).reshape(len(det_rows), -1)
    B = obs * W[None, :]                               # (F, 3M)
    C = B @ A                                          # (F, K)

    # Batched projected Gauss-Seidel over the K coordinates.
    Wt = np.zeros((len(det_rows), K), dtype=np.float64)
    for sweep in range(sweeps):
        max_delta = 0.0
        for k in range(K):
            resid_k = C[:, k] - Wt @ G_reg[:, k] + Wt[:, k] * g_diag[k]
            new_k = np.clip(resid_k / g_diag[k], 0.0, 1.0)
            step = np.abs(new_k - Wt[:, k]).max() if len(new_k) else 0.0
            max_delta = max(max_delta, float(step))
            Wt[:, k] = new_k
        if progress_cb is not None and sweep % 10 == 0:
            progress_cb(sweep, sweeps)
        if max_delta < tol:
            break

    # Residual diagnostics (unit: fraction of observation energy).
    fit = Wt @ A.T                                     # (F, 3M)
    num = np.linalg.norm(fit - B, axis=1)
    den = np.maximum(np.linalg.norm(B, axis=1), 1e-9)
    resid_ratio = num / den

    weights = np.zeros((n, K), dtype=np.float32)
    weights[det_rows] = Wt.astype(np.float32)

    diag = {
        "n_solved": int(len(det_rows)),
        "sweeps_run": int(sweep + 1),
        "resid_ratio_median": float(np.median(resid_ratio))
        if len(resid_ratio) else 1.0,
        "resid_ratio_p90": float(np.percentile(resid_ratio, 90))
        if len(resid_ratio) else 1.0,
        "saturated_frac": float((Wt > 0.995).mean()) if Wt.size else 0.0,
    }
    return weights, names, diag
