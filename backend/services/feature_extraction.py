from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import cv2
from pathlib import Path

# ============================================================================
# BASE CLASSES
# ============================================================================

class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extractors."""

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.get_feature_name()

    @abstractmethod
    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        """Extract features from the input image."""
        raise NotImplementedError

    @abstractmethod
    def get_feature_name(self) -> str:
        """Return the name of the feature."""
        raise NotImplementedError

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the dimensionality of the feature vector."""
        raise NotImplementedError

    @abstractmethod
    def get_category(self) -> str:
        """Return the category (form, texture, color)."""
        raise NotImplementedError

    def _load_image(self, image: Union[str, np.ndarray, Path]) -> np.ndarray:
        """
        Returns a grayscale image.
        - If input is path/str: loads as grayscale
        - If input is np.ndarray: safely handles 1ch, (H,W,1), 3ch, 4ch
        Output dtype: float32 (pipeline style)
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
            return img.astype(np.float32)

        if isinstance(image, np.ndarray):
            img = image

            # (H,W,1) -> (H,W)
            if img.ndim == 3 and img.shape[2] == 1:
                img = img[:, :, 0]

            # (H,W) already gray
            if img.ndim == 2:
                return img.astype(np.float32)

            # (H,W,3) or (H,W,4) -> gray
            if img.ndim == 3 and img.shape[2] in (3, 4):
                # If RGBA, drop alpha first
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                # NOTE: assuming BGR if coming from OpenCV; if it's RGB, it still works "okay" for grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return gray.astype(np.float32)

            raise ValueError(f"Unsupported ndarray shape for image: {img.shape}")

        raise TypeError(f"Unsupported image type: {type(image)}")

    def _load_color_image(self, image: Union[str, np.ndarray, Path]) -> np.ndarray:
        """Load image as BGR uint8."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
            return img
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            # If float, convert safely to uint8 range
            if image.dtype != np.uint8:
                img_u8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if img_u8.ndim == 2:
                    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
                return img_u8
            return image
        raise TypeError(f"Unsupported image type: {type(image)}")


class FormExtractor(BaseFeatureExtractor):
    """Base class for form-related extractors."""

    def __init__(self, contour_mode: str = "canny", name: Optional[str] = None):
        super().__init__(name)
        assert contour_mode in {"otsu", "canny"}
        self.contour_mode = contour_mode

    def get_category(self) -> str:
        return "form"

    def _extract_contour(self, image: Union[str, np.ndarray, Path]) -> Optional[np.ndarray]:
        img = self._load_image(image)

        # Ensure uint8 for OpenCV
        if img.dtype != np.uint8:
            img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img_u8 = img

        if self.contour_mode == "otsu":
            return self._extract_contour_otsu(img_u8)
        else:
            return self._extract_contour_canny(img_u8)

    def _extract_contour_otsu(self, img_u8: np.ndarray) -> Optional[np.ndarray]:
        _, binary = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        contour = largest.squeeze(axis=1)
        if contour.ndim != 2 or contour.shape[0] < 4:
            return None
        return contour

    def _extract_contour_canny(self, img_u8: np.ndarray) -> Optional[np.ndarray]:
        blur = cv2.GaussianBlur(img_u8, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        filled = cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        contour = largest.squeeze(axis=1)
        if contour.ndim != 2 or contour.shape[0] < 4:
            return None
        return contour


class TextureExtractor(BaseFeatureExtractor):
    """Base class for texture-related extractors."""

    def get_category(self) -> str:
        return "texture"

    def _preprocess_for_texture(self, image: Union[str, np.ndarray, Path]) -> np.ndarray:
        """Load grayscale image for texture analysis."""
        return self._load_image(image)


class ColorExtractor(BaseFeatureExtractor):
    """Base class for color-related extractors."""

    def get_category(self) -> str:
        return "color"


# ============================================================================
# FORM EXTRACTORS
# ============================================================================

class FourierDescriptorExtractor(FormExtractor):
    """Fourier descriptors for shape."""

    def __init__(
        self,
        n_coeff: int = 15,              # ✅ updated default
        contour_mode: str = "canny",    # ✅ updated default
        name: Optional[str] = None,
    ):
        super().__init__(contour_mode=contour_mode, name=name)
        self.n_coeff = n_coeff

    def get_feature_name(self) -> str:
        return "fourier"

    def get_feature_dim(self) -> int:
        return self.n_coeff

    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        contour = self._extract_contour(image)

        if contour is None or len(contour) < 4:
            vector = np.zeros(self.n_coeff, dtype=np.float32)
        else:
            z = contour[:, 0].astype(np.complex128) + 1j * contour[:, 1]
            z -= z.mean()
            F = np.fft.fft(z)

            denom = np.abs(F[1]) if np.abs(F[1]) > 1e-8 else 1.0
            desc = np.abs(F[1 : self.n_coeff + 1]) / denom

            vector = np.pad(
                desc,
                (0, max(0, self.n_coeff - len(desc))),
                "constant",
            ).astype(np.float32)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return {
            "vector": vector,
            "metadata": {
                "n_coeff": self.n_coeff,
                "contour_mode": self.contour_mode,
            },
            "name": self._name,
        }


class OrientationHistogramExtractor(FormExtractor):
    """Rotation-invariant, length-weighted histogram of contour orientations (unsigned 0–180°)."""

    def __init__(
        self,
        bins: int = 36,
        contour_mode: str = "canny",
        name: Optional[str] = None,
    ):
        super().__init__(contour_mode=contour_mode, name=name)
        self.bins = bins

    def get_feature_name(self) -> str:
        return "orientation"

    def get_feature_dim(self) -> int:
        return self.bins

    @staticmethod
    def _circular_align_peak(hist: np.ndarray) -> np.ndarray:
        """Rotate histogram so the dominant peak is at index 0 (rotation invariance)."""
        if hist.size == 0:
            return hist
        k = int(np.argmax(hist))
        return np.roll(hist, -k)

    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        contour = self._extract_contour(image)
        if contour is None or len(contour) < 2:
            vector = np.zeros(self.bins, dtype=np.float32)
        else:
            diffs = np.diff(contour, axis=0, append=contour[:1]).astype(np.float32)
            dx = diffs[:, 0]
            dy = diffs[:, 1]

            weights = np.sqrt(dx * dx + dy * dy)

            angles = np.arctan2(dy, dx)  # [-pi, pi]
            deg = (np.degrees(angles) + 360.0) % 360.0
            deg = deg % 180.0  # unsigned orientation [0,180)

            hist, _ = np.histogram(
                deg,
                bins=self.bins,
                range=(0.0, 180.0),
                weights=weights,
                density=False,
            )

            hist = hist.astype(np.float32)
            hist = self._circular_align_peak(hist)
            vector = hist

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return {
            "vector": vector.astype(np.float32),
            "metadata": {
                "bins": self.bins,
                "range_deg": [0, 180],
                "unsigned": True,
                "weighted_by": "segment_length",
                "alignment": "circular_peak",
                "contour_mode": self.contour_mode,
            },
            "name": self._name,
        }


# ============================================================================
# TEXTURE EXTRACTORS
# ============================================================================

class TamuraExtractor(TextureExtractor):
    """Tamura features: coarseness, contrast, directionality."""

    def __init__(self, kmax: int = 4, n_bins: int = 16, name: Optional[str] = None):
        super().__init__(name)
        self.kmax = kmax
        self.n_bins = n_bins

    def get_feature_name(self) -> str:
        return "tamura"

    def get_feature_dim(self) -> int:
        return 3

    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        img = self._preprocess_for_texture(image) / 255.0
        coarseness = self._coarseness(img)
        contrast = self._contrast(img)
        directionality = self._directionality(img)
        vector = np.array([coarseness, contrast, directionality], dtype=np.float32)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return {
            "vector": vector,
            "metadata": {"kmax": self.kmax, "n_bins": self.n_bins},
            "name": self._name,
        }

    # ---------------------------
    # FIX #1: Coarseness (classic per-pixel best scale)
    # ---------------------------
    def _coarseness(self, img: np.ndarray) -> float:
        """
        Classic Tamura coarseness:
        - Build multi-scale averaged images A_k using box filters of size 2^k
        - For each pixel, compute energies E_k based on |A_k(x+d,y)-A_k(x-d,y)| and |A_k(x,y+d)-A_k(x,y-d)|
        - Choose best scale k per pixel, then coarseness = mean(2^k_best)

        Notes:
        - Uses border wrapping via np.roll (consistent & fast).
        """
        h, w = img.shape
        kmax = max(1, int(self.kmax))

        # A_k: averaged images at each scale
        A = []
        for k in range(kmax):
            size = 2 ** k
            kernel = np.ones((size, size), dtype=np.float32) / float(size * size)
            A_k = cv2.filter2D(img, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
            A.append(A_k)
        A = np.stack(A, axis=0)  # (k, h, w)

        # Energies per scale (k, h, w)
        E = np.zeros((kmax, h, w), dtype=np.float32)

        for k in range(kmax):
            d = 2 ** k
            # symmetric differences (x +/- d) and (y +/- d)
            # using roll gives wrap; good enough for descriptor consistency
            Ah_p = np.roll(A[k], -d, axis=1)
            Ah_m = np.roll(A[k],  d, axis=1)
            Av_p = np.roll(A[k], -d, axis=0)
            Av_m = np.roll(A[k],  d, axis=0)

            Eh = np.abs(Ah_p - Ah_m)
            Ev = np.abs(Av_p - Av_m)

            E[k] = np.maximum(Eh, Ev)

        # Best scale per pixel
        k_best = np.argmax(E, axis=0).astype(np.int32)  # (h,w)

        # Coarseness is mean of the corresponding size
        S_best = (2.0 ** k_best).astype(np.float32)
        return float(np.mean(S_best))

    def _contrast(self, img: np.ndarray) -> float:
        mu4 = float(np.mean((img - img.mean()) ** 4))
        sigma2 = float(np.var(img))
        alpha4 = mu4 / (sigma2 ** 2 + 1e-8)
        return float((sigma2 ** 0.5) / (alpha4 ** 0.25 + 1e-8))
    # ---------------------------
    # FIX #2: Directionality (proper orientation + magnitude weighting)
    # ---------------------------
    def _directionality(self, img: np.ndarray) -> float:
        """
        Directionality:
        - Compute gradients gx, gy
        - Orientation (0..pi): theta = (atan2(gy,gx) + pi) % pi
        - Build histogram over theta, weighted by magnitude
        - Output "peakedness" (variance around mean) like your current style

        This fixes the 'abs()' quadrant-collapse issue.
        """
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

        mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        theta = (np.arctan2(gy, gx) + np.pi) % np.pi  # [0, pi)

        # threshold to ignore weak gradients (more stable than mean sometimes)
        # keep your "mean" spirit but slightly safer:
        thresh = float(np.mean(mag)) + 1e-8
        mask = mag > thresh

        if not np.any(mask):
            return 0.0

        hist, _ = np.histogram(
            theta[mask],
            bins=self.n_bins,
            range=(0.0, float(np.pi)),
            weights=mag[mask],
        )

        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)

        # peakedness (higher when strong dominant direction exists)
        return float(np.sum((hist - hist.mean()) ** 2))


class GaborExtractor(TextureExtractor):
    """Gabor filter bank features (mean + std per filter)."""

    def __init__(self, n_scales: int = 3, n_orientations: int = 4, name: Optional[str] = None):
        super().__init__(name)
        self.n_scales = n_scales
        self.n_orientations = n_orientations

    def get_feature_name(self) -> str:
        return "gabor"

    def get_feature_dim(self) -> int:
        return self.n_scales * self.n_orientations * 2

    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        img = self._preprocess_for_texture(image) / 255.0
        features = []
        for s in range(self.n_scales):
            lambda_ = 4 * (2 ** s)
            for o in range(self.n_orientations):
                theta = o * np.pi / self.n_orientations
                kernel = cv2.getGaborKernel((21, 21), 5, theta, lambda_, 0.5, 0, ktype=cv2.CV_32F)
                resp = cv2.filter2D(img, cv2.CV_32F, kernel)
                features.extend([float(resp.mean()), float(resp.std())])

        vector = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return {
            "vector": vector,
            "metadata": {"n_scales": self.n_scales, "n_orientations": self.n_orientations},
            "name": self._name,
        }


# ============================================================================
# COLOR EXTRACTORS
# ============================================================================

class HSVHistogramExtractor(ColorExtractor):
    """3D HSV histogram."""

    def __init__(self, h_bins: int = 8, sv_bins: int = 8, name: Optional[str] = None):
        super().__init__(name)
        self.h_bins = h_bins
        self.sv_bins = sv_bins

    def get_feature_name(self) -> str:
        return "hsv_histogram"

    def get_feature_dim(self) -> int:
        return self.h_bins * self.sv_bins * self.sv_bins

    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        img = self._load_color_image(image)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [self.h_bins, self.sv_bins, self.sv_bins],
            [0, 180, 0, 256, 0, 256],
        )
        hist = cv2.normalize(hist, hist).flatten()
        vector = hist.astype(np.float32)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return {
            "vector": vector,
            "metadata": {"h_bins": self.h_bins, "sv_bins": self.sv_bins},
            "name": self._name,
        }


# NOTE: Keeping this class (and return format) so nothing breaks if still imported somewhere.
class DominantColorsExtractor(ColorExtractor):
    """Dominant colors via k-means (Lab space)."""

    def __init__(self, n_colors: int = 5, name: Optional[str] = None):
        super().__init__(name)
        self.n_colors = n_colors

    def get_feature_name(self) -> str:
        return "dominant_colors"

    def get_feature_dim(self) -> int:
        return self.n_colors * 3

    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        img_bgr = self._load_color_image(image)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

        pixels = img_lab.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

        _, labels, centers = cv2.kmeans(
            pixels, self.n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )  # type: ignore

        counts = np.bincount(labels.flatten(), minlength=self.n_colors)
        order = np.argsort(-counts)
        centers = centers[order]

        L = centers[:, 0] / 255.0
        a = (centers[:, 1] - 128.0) / 127.0
        b = (centers[:, 2] - 128.0) / 127.0
        centers_norm = np.stack([L, a, b], axis=1)

        vector = centers_norm.flatten().astype(np.float32)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return {
            "vector": vector,
            "metadata": {"n_colors": self.n_colors},
            "name": self._name,
        }


# ============================================================================
# SERVICES
# ============================================================================

class FeatureExtractionService:
    """Service to extract and combine features from multiple extractors."""

    DEFAULT_WEIGHTS = {
        "form": {"fourier": 0.6, "orientation": 0.4},
        "texture": {"tamura": 0.5, "gabor": 0.5},
        "color": {"hsv_histogram": 1.0},
    }

    def __init__(
        self,
        extractors: List[BaseFeatureExtractor],
        weights: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.extractors = extractors
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._group_by_category()

    def _group_by_category(self):
        self.by_cat = {"form": [], "texture": [], "color": []}
        for e in self.extractors:
            self.by_cat[e.get_category()].append(e)

    def extract(
        self,
        image: Union[str, np.ndarray, Path],
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract features per category and combine weighted vectors."""
        if categories is None:
            categories = ["form", "texture", "color"]

        result: Dict[str, Any] = {}

        for cat in categories:
            feats: Dict[str, Any] = {}
            weighted_vectors: List[np.ndarray] = []

            exts = self.by_cat.get(cat, [])
            if not exts:
                continue

            for ext in exts:
                feat = ext.extract(image)
                name = ext.get_feature_name()
                feats[name] = feat

                # if weight not found, fallback to uniform inside that category
                weight = self.weights.get(cat, {}).get(name, 1.0 / max(1, len(exts)))
                weighted_vectors.append(feat["vector"] * float(weight))

            if feats and weighted_vectors:
                combined = np.concatenate(weighted_vectors).astype(np.float32)
                norm = np.linalg.norm(combined)
                if norm > 0:
                    combined /= norm
                feats["combined"] = combined
                result[cat] = feats

        return result


class SimilarityComputer:
    """Compute similarity between feature sets."""

    DEFAULT_WEIGHTS = {"form": 0.5, "texture": 0.3, "color": 0.2}

    def __init__(self, weights: Optional[Dict] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def compute(
        self,
        f1: Dict,
        f2: Dict,
        categories: Optional[List[str]] = None,
        metric: str = "euclidean",
    ) -> float:
        if categories is None:
            categories = list(set(f1.keys()) & set(f2.keys()))
        if not categories:
            return 0.0
        if metric not in ["cosine", "euclidean"]:
            raise ValueError("Metric must be 'cosine' or 'euclidean'.")

        sims: List[float] = []
        used_cats: List[str] = []

        for cat in categories:
            cat1, cat2 = f1.get(cat, {}), f2.get(cat, {})
            if "combined" not in cat1 or "combined" not in cat2:
                continue
            v1, v2 = cat1["combined"], cat2["combined"]

            if metric == "cosine":
                sim = float(np.dot(v1, v2))
            else:
                dist = float(np.linalg.norm(v1 - v2))
                sim = 1.0 / (1.0 + dist)

            sims.append(sim)
            used_cats.append(cat)

        if not sims:
            return 0.0

        total, total_w = 0.0, 0.0
        for cat, sim in zip(used_cats, sims):
            w = float(self.weights.get(cat, 1.0 / len(sims)))
            total += w * sim
            total_w += w

        return float(total / (total_w + 1e-12))

    def compute_with_class_filter(
        self,
        query_features: Dict,
        query_class: str,
        base_features: Dict[str, Dict],
        categories: Optional[List[str]] = None,
        metric: str = "euclidean",
        same_class_only: bool = False,
    ) -> List[Tuple[str, float]]:
        if categories is None:
            categories = ["form", "texture", "color"]

        similarities: List[Tuple[str, float]] = []

        for path, data in base_features.items():
            if data.get("num_objects", 0) == 0:
                continue

            objects = data.get("objects", [])
            if isinstance(objects, np.ndarray):
                objects = objects.tolist()

            max_sim = 0.0

            for obj in objects:
                obj_class = obj.get("class_name", "unknown")
                if same_class_only and obj_class != query_class:
                    continue
                try:
                    sim = self.compute(query_features, obj["features"], categories=categories, metric=metric)
                    if sim > max_sim:
                        max_sim = sim
                except Exception:
                    continue

            if max_sim > 0:
                similarities.append((path, float(max_sim)))

        return sorted(similarities, key=lambda x: x[1], reverse=True)
