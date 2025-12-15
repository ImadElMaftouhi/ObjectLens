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
        pass

    @abstractmethod
    def get_feature_name(self) -> str:
        """Return the name of the feature."""
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the dimensionality of the feature vector."""
        pass

    @abstractmethod
    def get_category(self) -> str:
        """Return the category (form, texture, color)."""
        pass

    def _load_image(self, image: Union[str, np.ndarray, Path]) -> np.ndarray:
        """Load image as grayscale float32."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
            return img.astype(np.float32)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            return image.astype(np.float32)
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
            return image
        raise TypeError(f"Unsupported image type: {type(image)}")


class FormExtractor(BaseFeatureExtractor):
    """Base class for form-related extractors."""

    def get_category(self) -> str:
        return "form"

    def _extract_contour(self, image: Union[str, np.ndarray, Path]) -> Optional[np.ndarray]:
        """Extract the largest external contour."""
        img = self._load_image(image)
        _, binary = cv2.threshold(img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        return largest.squeeze(axis=1)


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

    def __init__(self, n_coeff: int = 40, name: Optional[str] = None):
        super().__init__(name)
        self.n_coeff = n_coeff

    def get_feature_name(self) -> str:
        return "fourier"

    def get_feature_dim(self) -> int:
        return self.n_coeff

    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        """Extract normalized Fourier descriptors."""
        contour = self._extract_contour(image)
        if contour is None or len(contour) < 4:
            vector = np.zeros(self.n_coeff, dtype=np.float32)
        else:
            z = contour[:, 0].astype(np.complex128) + 1j * contour[:, 1]
            z -= z.mean()
            F = np.fft.fft(z)
            denom = np.abs(F[1]) if np.abs(F[1]) > 1e-8 else 1.0
            desc = np.abs(F[1:self.n_coeff + 1]) / denom
            vector = np.pad(desc, (0, max(0, self.n_coeff - len(desc))), "constant").astype(np.float32)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return {"vector": vector, "metadata": {"n_coeff": self.n_coeff}, "name": self._name}


class OrientationHistogramExtractor(FormExtractor):
    """Histogram of contour orientations."""

    def __init__(self, bins: int = 36, name: Optional[str] = None):
        super().__init__(name)
        self.bins = bins

    def get_feature_name(self) -> str:
        return "orientation"

    def get_feature_dim(self) -> int:
        return self.bins

    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        """Extract normalized orientation histogram."""
        contour = self._extract_contour(image)
        if contour is None or len(contour) < 2:
            vector = np.zeros(self.bins, dtype=np.float32)
        else:
            diffs = np.diff(contour, axis=0, append=contour[:1])
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            deg = (np.degrees(angles) + 360) % 360
            hist, _ = np.histogram(deg, bins=self.bins, range=(0, 360), density=True)
            vector = hist.astype(np.float32)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return {"vector": vector, "metadata": {"bins": self.bins}, "name": self._name}

# ============================================================================
# TEXTURE EXTRACTORS
# ============================================================================

class TamuraExtractor(TextureExtractor):
    """Tamura features: coarseness, contrast, directionality."""

    def __init__(self, kmax: int = 5, n_bins: int = 16, name: Optional[str] = None):
        super().__init__(name)
        self.kmax = kmax
        self.n_bins = n_bins

    def get_feature_name(self) -> str:
        return "tamura"

    def get_feature_dim(self) -> int:
        return 3

    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        """Extract and normalize Tamura features."""
        img = self._preprocess_for_texture(image) / 255.0
        coarseness = self._coarseness(img)
        contrast = self._contrast(img)
        directionality = self._directionality(img)
        vector = np.array([coarseness, contrast, directionality], dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return {"vector": vector, "metadata": {"kmax": self.kmax, "n_bins": self.n_bins}, "name": self._name}

    def _coarseness(self, img: np.ndarray) -> float:
        h, w = img.shape
        S = np.zeros((self.kmax, h, w))
        for k in range(self.kmax):
            size = 2 ** k
            kernel = np.ones((size, size)) / (size * size)
            S[k] = cv2.filter2D(img, -1, kernel)
        E = np.zeros((self.kmax * 2, h, w))
        for k in range(self.kmax):
            size = 2 ** k
            if size < w:
                E[k] = np.abs(S[k] - np.roll(S[k], size, axis=1))
            if size < h:
                E[k + self.kmax] = np.abs(S[k] - np.roll(S[k], size, axis=0))
        best_k = np.argmax(E.reshape(self.kmax * 2, -1), axis=0) % self.kmax
        return np.mean(2 ** best_k.reshape(h, w))

    def _contrast(self, img: np.ndarray) -> float:
        mu4 = np.mean((img - img.mean()) ** 4)
        sigma2 = np.var(img)
        alpha4 = mu4 / (sigma2 ** 2 + 1e-8)
        return sigma2 ** 0.5 / (alpha4 ** 0.25 + 1e-8)

    def _directionality(self, img: np.ndarray) -> float:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag = np.sqrt(gx**2 + gy**2)
        theta = np.arctan2(np.abs(gy), np.abs(gx)) % np.pi
        thresh = mag.mean()
        hist, _ = np.histogram(theta[mag > thresh], bins=self.n_bins, range=(0, np.pi))
        hist = hist / (hist.sum() + 1e-8)
        return np.sum((hist - hist.mean()) ** 2)


class GaborExtractor(TextureExtractor):
    """Gabor filter bank features (mean + std per filter)."""

    def __init__(self, n_scales: int = 4, n_orientations: int = 6, name: Optional[str] = None):
        super().__init__(name)
        self.n_scales = n_scales
        self.n_orientations = n_orientations

    def get_feature_name(self) -> str:
        return "gabor"

    def get_feature_dim(self) -> int:
        return self.n_scales * self.n_orientations * 2

    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        """Extract normalized Gabor responses."""
        img = self._preprocess_for_texture(image) / 255.0
        features = []
        for s in range(self.n_scales):
            lambda_ = 4 * (2 ** s)
            for o in range(self.n_orientations):
                theta = o * np.pi / self.n_orientations
                kernel = cv2.getGaborKernel((21, 21), 5, theta, lambda_, 0.5, 0, ktype=cv2.CV_32F)
                resp = cv2.filter2D(img, cv2.CV_32F, kernel)
                features.extend([resp.mean(), resp.std()])
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

    def __init__(self, h_bins: int = 4, sv_bins: int = 4, name: Optional[str] = None):
        super().__init__(name)
        self.h_bins = h_bins
        self.sv_bins = sv_bins

    def get_feature_name(self) -> str:
        return "hsv_histogram"

    def get_feature_dim(self) -> int:
        return self.h_bins * self.sv_bins * self.sv_bins

    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        """Extract normalized HSV histogram."""
        img = self._load_color_image(image)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [self.h_bins, self.sv_bins, self.sv_bins], [0, 180, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        vector = hist.astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return {"vector": vector, "metadata": {"h_bins": self.h_bins, "sv_bins": self.sv_bins}, "name": self._name}


class DominantColorsExtractor(ColorExtractor):
    """Dominant colors via k-means."""

    def __init__(self, n_colors: int = 5, name: Optional[str] = None):
        super().__init__(name)
        self.n_colors = n_colors

    def get_feature_name(self) -> str:
        return "dominant_colors"

    def get_feature_dim(self) -> int:
        return self.n_colors * 3

    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        """Extract normalized dominant colors (sorted by prevalence)."""
        img = self._load_color_image(image)
        pixels = img.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, labels, centers = cv2.kmeans(pixels, self.n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS) # type: ignore
        counts = np.bincount(labels.flatten(), minlength=self.n_colors)
        order = np.argsort(-counts)
        centers = centers[order] / 255.0
        vector = centers.flatten().astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return {"vector": vector, "metadata": {"n_colors": self.n_colors}, "name": self._name}

# ============================================================================
# SERVICES
# ============================================================================

class FeatureExtractionService:
    """Service to extract and combine features from multiple extractors."""

    DEFAULT_WEIGHTS = {
        "form": {"fourier": 0.6, "orientation": 0.4},
        "texture": {"tamura": 0.5, "gabor": 0.5},
        "color": {"hsv_histogram": 0.2, "dominant_colors": 0.8},
    }

    def __init__(
        self, extractors: List[BaseFeatureExtractor], weights: Optional[Dict[str, Dict[str, float]]] = None
    ):
        self.extractors = extractors
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._group_by_category()

    def _group_by_category(self):
        self.by_cat = {"form": [], "texture": [], "color": []}
        for e in self.extractors:
            self.by_cat[e.get_category()].append(e)

    def extract(self, image: Union[str, np.ndarray, Path], categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Extract features per category and combine weighted vectors."""
        if categories is None:
            categories = ["form", "texture", "color"]
        result = {}
        for cat in categories:
            feats = {}
            weighted_vectors = []
            for ext in self.by_cat.get(cat, []):
                feat = ext.extract(image)
                name = ext.get_feature_name()
                feats[name] = feat
                weight = self.weights.get(cat, {}).get(name, 1.0 / len(self.by_cat.get(cat, [])))
                weighted_vectors.append(feat["vector"] * weight)
            if feats and weighted_vectors:
                combined = np.concatenate(weighted_vectors)
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

    def compute(self, f1: Dict, f2: Dict, categories: Optional[List[str]] = None, metric: str = "euclidean") -> float:
        """
        Compute similarity between two feature sets.

        Parameters
        ----------
        f1 : dict
            Feature set 1.
        f2 : dict
            Feature set 2.
        categories : list, optional
            Categories to consider for similarity computation.
        metric : str, optional
            Metric to use for similarity computation. Must be 'cosine' or 'euclidean'.

        Returns
        -------
        float
            Similarity score between 0 and 1.
        """

        if categories is None:
            categories = list(set(f1.keys()) & set(f2.keys()))
        if not categories:
            return 0.0
        if metric not in ["cosine", "euclidean"]:
            raise ValueError("Metric must be 'cosine' or 'euclidean'.")

        sims = []
        for cat in categories:
            cat1, cat2 = f1.get(cat, {}), f2.get(cat, {})
            if "combined" not in cat1 or "combined" not in cat2:
                continue
            v1, v2 = cat1["combined"], cat2["combined"]
            if metric == "cosine":
                sim = np.dot(v1, v2)
            else:  # euclidean
                dist = np.linalg.norm(v1 - v2)
                sim = 1.0 / (1.0 + dist)
            sims.append(sim)

        if not sims:
            return 0.0

        total, total_w = 0.0, 0.0
        for cat, sim in zip(categories, sims):
            w = self.weights.get(cat, 1.0 / len(sims))
            total += w * sim
            total_w += w
        return total / total_w

    def compute_with_class_filter(self,
        query_features: Dict,query_class: str,
        base_features: Dict[str, Dict],
        categories: Optional[List[str]] = None, 
        metric: str = "euclidean",
        same_class_only: bool = False,
    ) -> List[Tuple[str, float]]:

        if categories is None:
            categories = ["form", "texture", "color"]
        
        similarities = []
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
                    sim = self.compute(query_features, obj["features"], metric=metric)
                    # use the object with highest similarity score
                    if sim > max_sim:
                        max_sim = sim
                except Exception:
                    continue
            if max_sim > 0:
                similarities.append((path, max_sim))
        return sorted(similarities, key=lambda x: x[1], reverse=True)