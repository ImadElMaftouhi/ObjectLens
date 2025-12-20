from __future__ import annotations

from typing import Any, Dict, Optional
import base64
import io

import numpy as np
import cv2
import matplotlib.pyplot as plt


def _png_b64_from_fig(fig) -> str:
    """Convert a matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _jpg_b64_from_bgr(img_bgr: np.ndarray, quality: int = 90) -> str:
    """Convert BGR image to base64 JPEG string."""
    if img_bgr is None or img_bgr.size == 0:
        return ""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode(".jpg", img_bgr, encode_params)
    if not ok:
        return ""
    return base64.b64encode(enc.tobytes()).decode("utf-8")


def build_query_descriptor_viz(
    crop_bgr: np.ndarray,
    q_feats: Dict[str, Any],
    *,
    include_crop_preview: bool = True,
) -> Dict[str, Any]:
    """
    Build meaningful, professor-friendly visualizations for the query object descriptors.

    Returns JSON-friendly dict:
      {
        "summaries": {...},
        "images_b64": {
            "crop_jpg": "...",
            "orientation_hist_png": "...",
            "fourier_png": "...",
            "tamura_png": "...",
            "hue_hist_png": "...",
            "sv_heatmap_png": "..."
        }
      }

    Notes:
    - Does NOT recompute features. Uses q_feats (already extracted in /topk).
    - Base64 strings are WITHOUT data URI prefix. Frontend can render as:
        src={`data:image/png;base64,${...}`}
    """
    images_b64: Dict[str, str] = {}
    summaries: Dict[str, Any] = {}

    # -----------------------------
    # 0) Crop preview (context)
    # -----------------------------
    if include_crop_preview:
        images_b64["crop_jpg"] = _jpg_b64_from_bgr(crop_bgr)

    # -----------------------------
    # 1) FORM: Fourier + Orientation
    # -----------------------------
    form = q_feats.get("form") or {}
    fourier = (form.get("fourier") or {}).get("vector", None)
    orient = (form.get("orientation") or {}).get("vector", None)

    if isinstance(fourier, np.ndarray):
        summaries.setdefault("form", {})["fourier"] = {
            "dim": int(fourier.size),
            "first_values": [float(x) for x in fourier[: min(8, fourier.size)]],
        }

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(1, fourier.size + 1), fourier, marker="o")
        ax.set_title("Fourier Descriptors (magnitudes)")
        ax.set_xlabel("Coefficient index")
        ax.set_ylabel("Value (L2-normalized)")
        ax.grid(True, alpha=0.3)
        images_b64["fourier_png"] = _png_b64_from_fig(fig)

    if isinstance(orient, np.ndarray):
        summaries.setdefault("form", {})["orientation"] = {
            "bins": int(orient.size),
            "peak_bin": int(np.argmax(orient)) if orient.size > 0 else 0,
        }

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(np.arange(orient.size), orient)
        ax.set_title("Contour Orientation Histogram (rotation-aligned)")
        ax.set_xlabel("Bin index")
        ax.set_ylabel("Value (L2-normalized)")
        ax.grid(True, axis="y", alpha=0.3)
        images_b64["orientation_hist_png"] = _png_b64_from_fig(fig)

    # -----------------------------
    # 2) TEXTURE: Tamura (interpretable)
    # -----------------------------
    tex = q_feats.get("texture") or {}
    tamura_vec = (tex.get("tamura") or {}).get("vector", None)

    if isinstance(tamura_vec, np.ndarray) and tamura_vec.size == 3:
        coarseness, contrast, directionality = [float(x) for x in tamura_vec.tolist()]
        summaries.setdefault("texture", {})["tamura"] = {
            "coarseness": coarseness,
            "contrast": contrast,
            "directionality": directionality,
        }

        fig = plt.figure()
        ax = fig.add_subplot(111)
        labels = ["Coarseness", "Contrast", "Directionality"]
        ax.bar(labels, [coarseness, contrast, directionality])
        ax.set_title("Tamura Texture Features")
        ax.set_ylabel("Value (L2-normalized)")
        ax.grid(True, axis="y", alpha=0.3)
        images_b64["tamura_png"] = _png_b64_from_fig(fig)

    # -----------------------------
    # 3) COLOR: HSV (meaningful: Hue + SV distribution)
    # -----------------------------
    color = q_feats.get("color") or {}
    hsv_vec = (color.get("hsv_histogram") or {}).get("vector", None)
    hsv_meta = (color.get("hsv_histogram") or {}).get("metadata", None)

    if isinstance(hsv_vec, np.ndarray) and isinstance(hsv_meta, dict):
        h_bins = int(hsv_meta.get("h_bins", 0))
        sv_bins = int(hsv_meta.get("sv_bins", 0))

        expected = h_bins * sv_bins * sv_bins
        if h_bins > 0 and sv_bins > 0 and hsv_vec.size == expected:
            hist3d = hsv_vec.reshape((h_bins, sv_bins, sv_bins))

            # Hue marginal: sum over S,V
            h_marg = hist3d.sum(axis=(1, 2))
            # SV heatmap: sum over H
            sv_map = hist3d.sum(axis=0)

            # Normalize for display (not to change underlying vectors)
            h_marg_disp = h_marg / (h_marg.max() + 1e-8)
            sv_disp = sv_map / (sv_map.max() + 1e-8)

            summaries.setdefault("color", {})["hsv_histogram"] = {
                "h_bins": h_bins,
                "sv_bins": sv_bins,
                "dominant_h_bin": int(np.argmax(h_marg)) if h_marg.size else 0,
            }

            # Hue histogram plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.bar(np.arange(h_bins), h_marg_disp)
            ax.set_title("Hue Distribution (H marginal)")
            ax.set_xlabel("Hue bin")
            ax.set_ylabel("Relative dominance")
            ax.grid(True, axis="y", alpha=0.3)
            images_b64["hue_hist_png"] = _png_b64_from_fig(fig)

            # SV heatmap plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.imshow(sv_disp, origin="lower", aspect="auto")
            ax.set_title("S–V Distribution (sum over H)")
            ax.set_xlabel("V bin")
            ax.set_ylabel("S bin")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            images_b64["sv_heatmap_png"] = _png_b64_from_fig(fig)

    return {"summaries": summaries, "images_b64": images_b64}
