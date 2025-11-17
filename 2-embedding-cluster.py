import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

import insightface
from insightface.app import FaceAnalysis

from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from collections import Counter
import csv

from collections import defaultdict

# ----------------------------
# CONFIG
# ----------------------------
IMAGE_FOLDER = r"D:\project\photo\json\m6_collected_images"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EMBED_SUFFIX = ".faces.npz"     # per-image cache with embeddings & metadata
SUMMARY_CSV = "cluster_counts.csv"
MEMBERS_CSV = "face_clusters.csv"

# DBSCAN params (good starting points; tune on your set)
DBSCAN_EPS = 0.35      # cosine distance threshold (smaller = stricter)
DBSCAN_MIN_SAMPLES = 3 # require at least this many samples to form a cluster

# ----------------------------
# INSIGHTFACE INIT
# ----------------------------
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
# Note: If embeddings come back None, you may need to force a model:
# e.g., app.prepare(ctx_id=0, det_size=(640,640), det_model="buffalo_l", rec_model="buffalo_l")

# ----------------------------
# UTILITIES
# ----------------------------
def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VALID_EXTS

def is_output_image(p: Path) -> bool:
    # Your annotated outputs look like "name_output.jpg"
    # Skip anything whose stem contains "_output"
    return "_output" in p.stem

def per_image_cache_path(img_path: Path) -> Path:
    # Save sidecar as "name.faces.npz" next to the image
    return img_path.with_suffix(EMBED_SUFFIX)

def read_image(path: Path):
    img = cv2.imread(str(path))
    return img

# ----------------------------
# STEP 1: Persist embeddings per image (skip *_output.*)
# ----------------------------
def extract_and_cache_embeddings(folder: Path):
    if not folder.exists():
        print(f"[ERROR] Folder does not exist: {folder}")
        return

    for p in folder.iterdir():
        if not is_image_file(p):
            continue
        if is_output_image(p):
            # Skip annotated outputs
            continue

        cache_path = per_image_cache_path(p)
        if cache_path.exists():
            # Already processed; skip to speed up re-runs
            print(f"[SKIP] Cache exists: {cache_path.name}")
            continue

        img = read_image(p)
        if img is None:
            print(f"[WARN] Failed to read image: {p}")
            continue

        faces = app.get(img)

        # Collect embeddings + lightweight metadata
        embs = []
        bboxes = []
        scores = []

        for f in faces:
            emb = getattr(f, "embedding", None)
            if emb is None:
                # No embedding available for this face; skip it
                continue
            embs.append(np.asarray(emb))
            bboxes.append(np.asarray(f.bbox))
            scores.append(float(getattr(f, "det_score", 0.0)))

        # Save only if we have at least one embedding
        if len(embs) > 0:
            embs_arr = np.vstack(embs)           # (N, D)
            bboxes_arr = np.vstack(bboxes)       # (N, 4)
            scores_arr = np.asarray(scores)      # (N,)
            np.savez_compressed(
                cache_path,
                embeddings=embs_arr,
                bboxes=bboxes_arr,
                scores=scores_arr,
                image_path=str(p)
            )
            print(f"[OK] {p.name}: {embs_arr.shape[0]} face(s) cached -> {cache_path.name}")
        else:
            # Write an empty marker so we don't retry endlessly
            np.savez_compressed(
                cache_path,
                embeddings=np.empty((0, 0)),
                bboxes=np.empty((0, 4)),
                scores=np.empty((0,)),
                image_path=str(p)
            )
            print(f"[OK] {p.name}: no embeddings found (saved empty cache)")

# ----------------------------
# STEP 2: Load all embeddings & cluster
# ----------------------------
# def load_all_embeddings(folder: Path):
#     """Return (X, meta) where:
#        - X is (M, D) embeddings
#        - meta is a list of dicts with per-face metadata
#     """
#     X_list = []
#     meta: List[Dict[str, Any]] = []

#     for p in folder.iterdir():
#         if p.is_file() and p.suffix == EMBED_SUFFIX:
#             data = np.load(p, allow_pickle=False)
#             embs = data["embeddings"]
#             bboxes = data["bboxes"]
#             scores = data["scores"]
#             img_path = Path(str(data["image_path"]))

#             if embs.size == 0:
#                 continue

#             for i in range(embs.shape[0]):
#                 X_list.append(embs[i])
#                 meta.append({
#                     "image_path": str(img_path),
#                     "face_index": i,
#                     "bbox": bboxes[i].tolist(),
#                     "det_score": float(scores[i]),
#                     "cache_file": str(p)
#                 })

#     if len(X_list) == 0:
#         return np.empty((0, 0)), meta

#     X = np.vstack(X_list)
#     return X, meta

def load_all_embeddings(folder: Path):
    """Return (X, meta) where:
       - X is (M, D) embeddings
       - meta is a list of dicts with per-face metadata
    """
    X_list = []
    meta = []

    cache_count = 0
    face_rows = 0

    for p in folder.iterdir():
        if p.is_file() and p.name.endswith(EMBED_SUFFIX):  # <— FIXED
            cache_count += 1
            data = np.load(p, allow_pickle=False)
            embs = data["embeddings"]
            bboxes = data["bboxes"]
            scores = data["scores"]
            img_path = Path(str(data["image_path"]))

            # skip empty caches
            if embs.size == 0:
                continue

            for i in range(embs.shape[0]):
                X_list.append(embs[i])
                meta.append({
                    "image_path": str(img_path),
                    "face_index": i,
                    "bbox": bboxes[i].tolist(),
                    "det_score": float(scores[i]),
                    "cache_file": str(p)
                })
                face_rows += 1

    print(f"[LOAD] Found {cache_count} cache files, {face_rows} faces with embeddings.")
    if len(X_list) == 0:
        return np.empty((0, 0)), meta

    X = np.vstack(X_list)
    return X, meta


def cluster_embeddings(X: np.ndarray, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES):
    """Cluster with DBSCAN using cosine distance. Returns labels (M,)."""
    if X.size == 0:
        return np.array([], dtype=int)

    # Normalize to unit length (best practice before cosine-distance clustering)
    Xn = normalize(X, norm="l2")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(Xn)
    return labels

# ----------------------------
# STEP 3: Count appearances and save reports
# ----------------------------
def save_reports(labels: np.ndarray, meta: List[Dict[str, Any]], out_dir: Path):
    """Writes:
       - cluster_counts.csv (cluster_id, count)
       - face_clusters.csv (one row per face with cluster id + metadata)
    """
    out_counts = out_dir / SUMMARY_CSV
    out_members = out_dir / MEMBERS_CSV

    # Count clusters (exclude noise label = -1)
    label_list = labels.tolist()
    counts = Counter(l for l in label_list if l != -1)
    # Sort by count desc
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # Save counts
    with open(out_counts, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "count"])
        for cid, cnt in sorted_counts:
            writer.writerow([cid, cnt])

    # Save full membership
    with open(out_members, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "image_path", "face_index", "det_score", "bbox"])
        for lbl, m in zip(labels, meta):
            writer.writerow([lbl, m["image_path"], m["face_index"], m["det_score"], m["bbox"]])

    print(f"[REPORT] Wrote: {out_counts.name} and {out_members.name}")
    if sorted_counts:
        top = sorted_counts[:10]
        print("[TOP CLUSTERS] (cluster_id, count):", top)
    else:
        print("[INFO] No clusters found (or all were labeled as noise -1).")

# ---------------------------
# 🆕 New helper functions for face crops
# ---------------------------
# ----------------------------
# FACE CROP UTILS
# ----------------------------
def _clip_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

def _expand_bbox(bbox: np.ndarray, img_w: int, img_h: int, margin_ratio: float = 0.20, make_square: bool = False):
    """bbox is [x1, y1, x2, y2] (floats). Add margin and optionally square it, then clip."""
    x1, y1, x2, y2 = map(float, bbox.tolist())
    bw, bh = (x2 - x1), (y2 - y1)
    # add margin
    mx, my = bw * margin_ratio, bh * margin_ratio
    x1 -= mx; y1 -= my; x2 += mx; y2 += my
    # make square around center (optional)
    if make_square:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1)
        x1, x2 = cx - side / 2.0, cx + side / 2.0
        y1, y2 = cy - side / 2.0, cy + side / 2.0
    return _clip_bbox(x1, y1, x2, y2, img_w, img_h)

def crop_face_from_image(img: np.ndarray, bbox: np.ndarray, margin_ratio: float = 0.20, make_square: bool = True, resize_to: Optional[int] = 256) -> np.ndarray:
    """Return a cropped face region as an image (BGR)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _expand_bbox(bbox, w, h, margin_ratio=margin_ratio, make_square=make_square)
    face = img[y1:y2, x1:x2].copy()
    if resize_to is not None:
        face = cv2.resize(face, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    return face

def export_faces_by_cluster(
    labels: np.ndarray,
    meta: list,
    output_root: Path,
    include_noise: bool = True,
    margin_ratio: float = 0.20,
    make_square: bool = True,
    resize_to: Optional[int] = 256
):
    """
    Saves cropped faces into cluster folders under output_root / "clusters".
    File name format: <orig_stem>_f<face_index>_cid<label>_<n>.jpg
    """
    cluster_dir = output_root / "clusters"
    cluster_dir.mkdir(exist_ok=True, parents=True)

    label_to_indices = defaultdict(list)
    for i, lbl in enumerate(labels):
        if lbl == -1 and not include_noise:
            continue
        label_to_indices[lbl].append(i)

    # Pre-load and cache images per path to avoid re-reading same file repeatedly
    img_cache = {}

    total_saved = 0
    for lbl, idxs in label_to_indices.items():
        sub = "noise" if lbl == -1 else f"ID_{lbl}"
        out_dir = cluster_dir / sub
        out_dir.mkdir(exist_ok=True, parents=True)

        for k, i in enumerate(idxs):
            m = meta[i]
            img_path = Path(m["image_path"])
            bbox = np.array(m["bbox"], dtype=float)
            face_index = m["face_index"]

            # read original image (BGR)
            if img_path not in img_cache:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"[WARN] Cannot read {img_path}, skipping.")
                    img_cache[img_path] = None
                else:
                    img_cache[img_path] = img
            img = img_cache[img_path]
            if img is None:
                continue

            # crop
            face_img = crop_face_from_image(
                img, bbox,
                margin_ratio=margin_ratio,
                make_square=make_square,
                resize_to=resize_to
            )

            # write
            stem = img_path.stem
            fout = out_dir / f"{stem}_f{face_index}_cid{lbl}_{k}.jpg"
            ok = cv2.imwrite(str(fout), face_img)
            if ok:
                total_saved += 1
            else:
                print(f"[WARN] Failed to save {fout}")

    print(f"[EXPORT] Saved {total_saved} cropped faces into {cluster_dir}")


# ----------------------------
# MAIN
# ----------------------------
def main():
    folder = Path(IMAGE_FOLDER)

    print("\n[1/4] Extract & cache embeddings per image (skips *_output.*)")
    extract_and_cache_embeddings(folder)

    print("\n[2/4] Load all embeddings & cluster")
    X, meta = load_all_embeddings(folder)
    if X.size == 0:
        print("[INFO] No embeddings to cluster. Check if recognition model is loading and faces are detected.")
        return

    labels = cluster_embeddings(X, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)

    print("\n[3/4] Count appearances & save reports")
    save_reports(labels, meta, folder)

    # 🆕 Step 4: Export cropped faces grouped by cluster
    print("\n[4/4] Export cropped faces grouped by cluster")
    export_faces_by_cluster(
        labels, meta, folder,
        include_noise=True,       # also export noise (-1)
        margin_ratio=0.20,        # enlarge box 20%
        make_square=True,         # make face crops square
        resize_to=256             # output 256x256 faces
    )

if __name__ == "__main__":
    main()
