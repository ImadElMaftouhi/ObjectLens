"""
Modular Feature Extraction System

This module provides a flexible, extensible framework for extracting various
image features organized into three categories: forms, textures, and colors.

Architecture:
    BaseFeatureExtractor (abstract)
        ├── FormExtractor (abstract, category='form')
        │   ├── FourierDescriptorExtractor
        │   ├── OrientationHistogramExtractor
        │   └── (future: HuMomentsExtractor, etc.)
        │
        ├── TextureExtractor (abstract, category='texture')
        │   ├── TamuraExtractor
        │   ├── GaborExtractor
        │   └── (future: LBPExtractor, etc.)
        │
        └── ColorExtractor (abstract, category='color')
            ├── HSVHistogramExtractor
            ├── DominantColorsExtractor
            └── (future: ColorMomentsExtractor, etc.)

Usage:
    # Initialize extractors
    fourier = FourierDescriptorExtractor(n_coeff=40)
    orientation = OrientationHistogramExtractor(bins=36)
    tamura = TamuraExtractor()
    gabor = GaborExtractor()
    hsv = HSVHistogramExtractor(bins=256)
    dominant = DominantColorsExtractor(n_colors=5)
    
    # Create service
    service = FeatureExtractionService([fourier, orientation, tamura, gabor, hsv, dominant])
    
    # Extract features (all categories by default)
    features = service.extract('image.jpg')
    
    # Or select specific categories
    features = service.extract('image.jpg', categories=['form', 'texture'])
    
    # Compute similarity
    computer = SimilarityComputer()
    similarity = computer.compute(features1, features2, selected_categories=['form', 'texture'])
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import cv2
from pathlib import Path


# ============================================================================
# BASE CLASSES
# ============================================================================

class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors.
    
    Each feature extractor must implement:
    - extract(): Extract features from an image
    - get_feature_name(): Return a unique name for this extractor
    - get_feature_dim(): Return the dimension of the feature vector
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the feature extractor.
        
        Args:
            name: Optional custom name. If not provided, uses get_feature_name()
        """
        self._name = name or self.get_feature_name()

    @abstractmethod
    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        """
        Extract features from an image.
        
        Args:
            image: Path to image file, or numpy array (BGR or RGB)
            
        Returns:
            Dictionary containing:
            - 'vector': numpy array of feature vector (normalized within category)
            - 'metadata': dict with extractor-specific settings
            - 'name': name of the feature extractor
        """
        pass
    
    @abstractmethod
    def get_feature_name(self) -> str:
        """
        Return a unique name for this feature extractor.
        
        Returns:
            String identifier (e.g., 'fourier', 'tamura', 'hsv_histogram')
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        Return the dimension of the feature vector.
        
        Returns:
            Integer dimension
        """
        pass
    
    @abstractmethod
    def get_category(self) -> str:
        """
        Return the category of this extractor.
        
        Returns:
            'form', 'texture', or 'color'
        """
        pass
    
    def _load_image(self, image: Union[str, np.ndarray, Path]) -> np.ndarray:
        """
        Helper method to load image from path or return array.
        
        Args:
            image: Path to image or numpy array
            
        Returns:
            Grayscale image as numpy array (float32, 0-255)
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)  
            if img is None:
                raise ValueError(f"Could not load image: {image}")
            return img.astype(np.float32)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                # Convert BGR/RGB to grayscale
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            return image.astype(np.float32)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def _load_color_image(self, image: Union[str, np.ndarray, Path]) -> np.ndarray:
        """
        Helper method to load color image.
        
        Args:
            image: Path to image or numpy array
            
        Returns:
            Color image as numpy array (BGR format)
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
            return img
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                # Convert grayscale to BGR
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")


class FormExtractor(BaseFeatureExtractor):
    """
    Abstract base class for form/shape feature extractors.
    Provides category-specific helper methods for contour extraction.
    """
    
    def get_category(self) -> str:
        return 'form'
    
    def _extract_contour(self, image: Union[str, np.ndarray, Path], 
                        verbose: int = 0) -> Optional[np.ndarray]:
        """
        Extract the largest contour from a binary image.
        
        Args:
            image: Path to image or numpy array
            verbose: Verbosity level
            
        Returns:
            Contour as numpy array (N, 2) or None if no contour found
        """
        img = self._load_image(image)
        
        # Binarize using Otsu threshold
        _, binary = cv2.threshold(img.astype(np.uint8), 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_NONE)
        
        if verbose:
            print(f"Found {len(contours)} contours")
        
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=lambda c: c.shape[0])
        return largest.squeeze()  # Remove dimensions of size 1


class TextureExtractor(BaseFeatureExtractor):
    """
    Abstract base class for texture feature extractors.
    Provides category-specific helper methods for texture preprocessing.
    """
    
    def get_category(self) -> str:
        return 'texture'
    
    def _preprocess_for_texture(self, image: Union[str, np.ndarray, Path]) -> np.ndarray:
        """
        Preprocess image for texture analysis (grayscale, float32).
        
        Args:
            image: Path to image or numpy array
            
        Returns:
            Preprocessed grayscale image (float32)
        """
        return self._load_image(image)


class ColorExtractor(BaseFeatureExtractor):
    """
    Abstract base class for color feature extractors.
    Provides category-specific helper methods for color space conversion.
    """
    
    def get_category(self) -> str:
        return 'color'
    
    def _convert_to_hsv(self, image: Union[str, np.ndarray, Path]) -> np.ndarray:
        """
        Convert image to HSV color space.
        
        Args:
            image: Path to image or numpy array
            
        Returns:
            HSV image as numpy array
        """
        img = self._load_color_image(image)
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# ============================================================================
# FORM EXTRACTORS
# ============================================================================

class FourierDescriptorExtractor(FormExtractor):
    """
    Extract Fourier descriptors for shape representation.
    Translation, scale, and rotation invariant.
    """
    
    def __init__(self, n_coeff: int = 40, name: Optional[str] = None):
        """
        Args:
            n_coeff: Number of Fourier coefficients to extract
            name: Optional custom name
        """
        super().__init__(name)
        self.n_coeff = n_coeff
    
    def get_feature_name(self) -> str:
        return 'fourier'
    
    def get_feature_dim(self) -> int:
        return self.n_coeff
    
    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        contour = self._extract_contour(image)
        
        if contour is None or len(contour) == 0:
            vector = np.zeros(self.n_coeff, dtype=np.float32)
        else:
            # Ensure shape (N, 2)
            contour = np.array(contour)
            if contour.ndim == 1:
                # Handle edge case
                vector = np.zeros(self.n_coeff, dtype=np.float32)
            else:
                # Convert to complex representation
                z = contour[:, 0].astype(np.float64) + 1j * contour[:, 1].astype(np.float64)
                m = len(z)
                
                # Translation invariance
                z = z - z.mean()
                
                # FFT
                F = np.fft.fft(z)
                
                # Scale invariance: divide by magnitude of first non-zero coefficient
                denom = np.abs(F[1]) if m > 1 and np.abs(F[1]) > 1e-8 else np.linalg.norm(F)
                if denom == 0:
                    denom = 1.0
                
                # Keep first n_coeff coefficients
                coeffs = F[:self.n_coeff]
                desc = np.abs(coeffs) / denom
                
                # Pad if necessary
                if len(desc) < self.n_coeff:
                    desc = np.pad(desc, (0, self.n_coeff - len(desc)), 'constant')
                
                vector = np.real(desc).astype(np.float32)
        
        # L2 normalize within category
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return {
            'vector': vector,
            'metadata': {
                'n_coeff': self.n_coeff,
                'normalization': 'L2'
            },
            'name': self._name
        }


class OrientationHistogramExtractor(FormExtractor):
    """
    Extract orientation histogram from contour edge directions.
    """
    
    def __init__(self, bins: int = 36, name: Optional[str] = None):
        """
        Args:
            bins: Number of bins for orientation histogram
            name: Optional custom name
        """
        super().__init__(name)
        self.bins = bins
    
    def get_feature_name(self) -> str:
        return 'orientation'
    
    def get_feature_dim(self) -> int:
        return self.bins
    
    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        contour = self._extract_contour(image)
        
        if contour is None or len(contour) < 2:
            vector = np.zeros(self.bins, dtype=np.float32)
        else:
            # Ensure shape (N, 2)
            contour = np.array(contour)
            if contour.ndim == 1:
                vector = np.zeros(self.bins, dtype=np.float32)
            else:
                # Compute angles between consecutive points
                diffs = np.diff(contour, axis=0)
                angles = np.arctan2(diffs[:, 1], diffs[:, 0])  # radians -pi..pi
                
                # Convert to degrees 0..360
                deg = (np.degrees(angles) + 360) % 360
                
                # Create histogram
                hist, _ = np.histogram(deg, bins=self.bins, range=(0, 360), density=True)
                vector = hist.astype(np.float32)
        
        # L2 normalize within category
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return {
            'vector': vector,
            'metadata': {
                'bins': self.bins,
                'normalization': 'L2'
            },
            'name': self._name
        }


# ============================================================================
# TEXTURE EXTRACTORS
# ============================================================================

class TamuraExtractor(TextureExtractor):
    """
    Extract Tamura texture features: Coarseness, Contrast, Directionality, Granularity
    """
    
    def __init__(self, kmax: int = 4, n_bins: int = 16, name: Optional[str] = None):
        """
        Args:
            kmax: Maximum scale for coarseness computation
            n_bins: Number of bins for directionality histogram
            name: Optional custom name
        """
        super().__init__(name)
        self.kmax = kmax
        self.n_bins = n_bins
    
    def get_feature_name(self) -> str:
        return 'tamura'
    
    def get_feature_dim(self) -> int:
        return 4  # coarseness, contrast, directionality, granularity
    
    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        img = self._preprocess_for_texture(image)
        
        coarseness = self._coarseness(img)
        contrast = self._contrast(img)
        directionality = self._directionality(img)
        granularity = self._granularity(img)
        
        vector = np.array([coarseness, contrast, directionality, granularity], 
                         dtype=np.float32)
        
        # L2 normalize within category
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return {
            'vector': vector,
            'metadata': {
                'kmax': self.kmax,
                'n_bins': self.n_bins,
                'normalization': 'L2'
            },
            'name': self._name
        }
    
    def _coarseness(self, img: np.ndarray) -> float:
        """Compute Tamura coarseness."""
        h, w = img.shape
        img_norm = img / 255.0
        
        averages = []
        for k in range(self.kmax):
            size = max(1, 2 ** k)
            avg = cv2.blur(img_norm, (size, size))
            averages.append(avg)
        
        best_scale = np.zeros_like(img_norm, dtype=np.float32)
        scale_map = np.full_like(img_norm, 1.0, dtype=np.float32)
        
        for k in range(self.kmax):
            size = 2 ** k
            avg = averages[k]
            
            diff_h = np.zeros_like(avg)
            diff_v = np.zeros_like(avg)
            
            if size < w:
                diff_h[:, :-size] = np.abs(avg[:, size:] - avg[:, :-size])
            if size < h:
                diff_v[:-size, :] = np.abs(avg[size:, :] - avg[:-size, :])
            
            diff = np.maximum(diff_h, diff_v)
            mask = diff > best_scale
            best_scale[mask] = diff[mask]
            scale_map[mask] = 2 ** k
        
        return float(np.mean(scale_map))
    
    def _contrast(self, img: np.ndarray) -> float:
        """Compute Tamura contrast."""
        img_norm = img / 255.0
        mu = np.mean(img_norm)
        diff = img_norm - mu
        sigma = np.sqrt(np.mean(diff ** 2)) + 1e-8
        m4 = np.mean(diff ** 4) + 1e-8
        alpha4 = m4 / (sigma ** 4)
        return float(sigma / (alpha4 ** 0.25 + 1e-8))
    
    def _directionality(self, img: np.ndarray) -> float:
        """Compute Tamura directionality."""
        img_norm = img / 255.0
        
        gx = cv2.Sobel(img_norm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_norm, cv2.CV_32F, 0, 1, ksize=3)
        
        mag = np.sqrt(gx ** 2 + gy ** 2)
        theta = np.arctan2(gy, gx)
        theta = (theta + np.pi) % np.pi
        
        thresh = 0.1 * np.max(mag)
        mask = mag > thresh
        
        if np.count_nonzero(mask) == 0:
            return 0.0
        
        angles = theta[mask]
        hist, _ = np.histogram(angles, bins=self.n_bins, range=(0, np.pi))
        hist = hist.astype(np.float32)
        if hist.sum() != 0:
            hist /= hist.sum()
        
        uniform = 1.0 / self.n_bins
        return float(np.sum((hist - uniform) ** 2))
    
    def _granularity(self, img: np.ndarray) -> float:
        """Compute Tamura granularity."""
        img_norm = img / 255.0
        lap = cv2.Laplacian(img_norm, cv2.CV_32F, ksize=3)
        return float(np.mean(np.abs(lap)))


class GaborExtractor(TextureExtractor):
    """
    Extract Gabor filter bank features for texture analysis.
    """
    
    def __init__(self, n_scales: int = 3, n_orientations: int = 4, 
                 name: Optional[str] = None):
        """
        Args:
            n_scales: Number of scales (wavelengths)
            n_orientations: Number of orientations
            name: Optional custom name
        """
        super().__init__(name)
        self.n_scales = n_scales
        self.n_orientations = n_orientations
        self.lambdas = [4, 8, 16][:n_scales]
        self.thetas = [k * np.pi / n_orientations for k in range(n_orientations)]
    
    def get_feature_name(self) -> str:
        return 'gabor'
    
    def get_feature_dim(self) -> int:
        return self.n_scales * self.n_orientations
    
    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        img = self._preprocess_for_texture(image)
        
        features = []
        for lam in self.lambdas:
            sigma = 0.56 * lam
            gamma = 0.5
            psi = 0
            ksize = int(4 * lam + 1)
            if ksize % 2 == 0:
                ksize += 1
            
            for theta in self.thetas:
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lam, gamma, psi, ktype=cv2.CV_32F
                )
                response = cv2.filter2D(img, cv2.CV_32F, kernel)
                energy = np.mean(response ** 2)
                features.append(energy)
        
        vector = np.array(features, dtype=np.float32)
        
        # L2 normalize within category
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return {
            'vector': vector,
            'metadata': {
                'n_scales': self.n_scales,
                'n_orientations': self.n_orientations,
                'normalization': 'L2'
            },
            'name': self._name
        }


# ============================================================================
# COLOR EXTRACTORS
# ============================================================================

class HSVHistogramExtractor(ColorExtractor):
    """
    Extract HSV color histogram features.
    """
    
    def __init__(self, bins: int = 256, normalize: bool = True, 
                 name: Optional[str] = None):
        """
        Args:
            bins: Number of bins per channel (H, S, V)
            normalize: Whether to normalize histogram
            name: Optional custom name
        """
        super().__init__(name)
        self.bins = bins
        self.normalize = normalize
    
    def get_feature_name(self) -> str:
        return 'hsv_histogram'
    
    def get_feature_dim(self) -> int:
        return 3 * self.bins  # H, S, V channels
    
    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        img = self._load_color_image(image)
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Compute histograms for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [self.bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [self.bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [self.bins], [0, 256])
        
        # Normalize if requested
        if self.normalize:
            hist_h = hist_h / (hist_h.sum() + 1e-8)
            hist_s = hist_s / (hist_s.sum() + 1e-8)
            hist_v = hist_v / (hist_v.sum() + 1e-8)
        
        # Concatenate into single vector
        vector = np.concatenate([hist_h.flatten(), hist_s.flatten(), 
                                hist_v.flatten()]).astype(np.float32)
        
        # L2 normalize within category
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return {
            'vector': vector,
            'metadata': {
                'bins': self.bins,
                'normalize': self.normalize,
                'normalization': 'L2'
            },
            'name': self._name
        }


class DominantColorsExtractor(ColorExtractor):
    """
    Extract dominant colors using K-means clustering.
    """
    
    def __init__(self, n_colors: int = 5, name: Optional[str] = None):
        """
        Args:
            n_colors: Number of dominant colors to extract
            name: Optional custom name
        """
        super().__init__(name)
        self.n_colors = n_colors
    
    def get_feature_name(self) -> str:
        return 'dominant_colors'
    
    def get_feature_dim(self) -> int:
        return self.n_colors * 3  # RGB values for each color
    
    def extract(self, image: Union[str, np.ndarray, Path]) -> Dict[str, Any]:
        img = self._load_color_image(image)
        
        # Reshape image to be a list of pixels
        pixels = img.reshape(-1, 3).astype(np.float32)
        
        # Apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, self.n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Sort by frequency (most common first)
        unique, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        dominant_colors = centers[sorted_indices].flatten()
        
        vector = dominant_colors.astype(np.float32)
        
        # Normalize to [0, 1]
        vector = vector / 255.0
        
        # L2 normalize within category
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return {
            'vector': vector,
            'metadata': {
                'n_colors': self.n_colors,
                'normalization': 'L2'
            },
            'name': self._name
        }


# ============================================================================
# FEATURE EXTRACTION SERVICE
# ============================================================================

class FeatureExtractionService:
    """
    Manages multiple feature extractors and combines their outputs by category.
    """
    
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        """
        Initialize with a list of feature extractors.
        
        Args:
            extractors: List of BaseFeatureExtractor instances
            
        Raises:
            ValueError: If duplicate extractor names found
        """
        self.extractors = extractors
        self._validate_extractors()
        self._group_by_category()
    
    def _validate_extractors(self):
        """Validate that all extractors have unique names."""
        names = [ext.get_feature_name() for ext in self.extractors]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate feature extractor names found!")
    
    def _group_by_category(self):
        """Group extractors by category."""
        self._extractors_by_category = {
            'form': [],
            'texture': [],
            'color': []
        }
        
        for ext in self.extractors:
            category = ext.get_category()
            if category not in self._extractors_by_category:
                raise ValueError(f"Unknown category: {category}")
            self._extractors_by_category[category].append(ext)
    
    def extract(self, image: Union[str, np.ndarray, Path], 
                categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract features using selected categories.
        
        Args:
            image: Path to image or numpy array
            categories: List of categories to extract. If None, extracts all categories.
                       Valid values: 'form', 'texture', 'color'
        
        Returns:
            Dictionary with structure:
            {
                'form': {
                    'fourier': {'vector': ..., 'metadata': ..., 'name': ...},
                    'orientation': {...},
                    'combined': np.array([...])  # concatenated and normalized
                },
                'texture': {...},
                'color': {...}
            }
        """
        if categories is None:
            categories = ['form', 'texture', 'color']
        
        # Validate categories
        valid_categories = {'form', 'texture', 'color'}
        invalid = set(categories) - valid_categories
        if invalid:
            raise ValueError(f"Invalid categories: {invalid}")
        
        result = {}
        
        for category in categories:
            extractors = self._extractors_by_category[category]
            
            if not extractors:
                # No extractors for this category, skip
                continue
            
            category_features = {}
            vectors = []
            
            for extractor in extractors:
                feature_result = extractor.extract(image)
                name = feature_result['name']
                category_features[name] = feature_result
                vectors.append(feature_result['vector'])
            
            # Create combined vector for this category
            if vectors:
                combined = np.concatenate(vectors)
                # Normalize combined vector within category
                norm = np.linalg.norm(combined)
                if norm > 0:
                    combined = combined / norm
                category_features['combined'] = combined
            
            result[category] = category_features
        
        return result
    
    def get_extractors_by_category(self, category: str) -> List[BaseFeatureExtractor]:
        """Get all extractors for a specific category."""
        return self._extractors_by_category.get(category, [])
    
    def get_extractor(self, name: str) -> Optional[BaseFeatureExtractor]:
        """Get extractor by name."""
        for ext in self.extractors:
            if ext.get_feature_name() == name:
                return ext
        return None


# ============================================================================
# SIMILARITY COMPUTER
# ============================================================================

class SimilarityComputer:
    """
    Computes weighted similarity between feature sets using two-level weighting:
    - Level 1: Extractor weights within each category
    - Level 2: Category weights across categories
    """
    
    # Default weights: extractor weights within category sum to 1.0
    # Category weights sum to 1.0
    DEFAULT_WEIGHTS = {
        'form': {
            'fourier': 0.6,
            'orientation': 0.4
        },
        'texture': {
            'tamura': 0.5,
            'gabor': 0.5
        },
        'color': {
            'hsv_histogram': 0.5,
            'dominant_colors': 0.5
        },
        'category_weights': {
            'form': 0.33,
            'texture': 0.34,
            'color': 0.33
        }
    }
    
    def __init__(self, weights: Optional[Dict[str, Any]] = None):
        """
        Initialize similarity computer with weights.
        
        Args:
            weights: Weight configuration. If None, uses DEFAULT_WEIGHTS.
                    Structure:
                    {
                        'form': {'fourier': 0.6, 'orientation': 0.4},
                        'texture': {'tamura': 0.5, 'gabor': 0.5},
                        'color': {'hsv_histogram': 0.5, 'dominant_colors': 0.5},
                        'category_weights': {'form': 0.33, 'texture': 0.34, 'color': 0.33}
                    }
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()
    
    def _validate_weights(self):
        """Validate that weights sum to 1.0 within categories and for category weights."""
        # Validate extractor weights within each category
        for category in ['form', 'texture', 'color']:
            if category in self.weights:
                cat_weights = self.weights[category]
                total = sum(cat_weights.values())
                if abs(total - 1.0) > 1e-6:
                    raise ValueError(f"Extractor weights in '{category}' must sum to 1.0, got {total}")
        
        # Validate category weights
        if 'category_weights' in self.weights:
            cat_total = sum(self.weights['category_weights'].values())
            if abs(cat_total - 1.0) > 1e-6:
                raise ValueError(f"Category weights must sum to 1.0, got {cat_total}")
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First feature vector
            vec2: Second feature vector
            
        Returns:
            Cosine similarity in [0, 1] (assuming normalized vectors)
        """
        dot_product = np.dot(vec1, vec2)
        # For normalized vectors, cosine similarity = dot product
        # Clamp to [0, 1] for safety
        return max(0.0, min(1.0, dot_product))
    
    def compute(self, features1: Dict[str, Any], features2: Dict[str, Any],
                selected_categories: Optional[List[str]] = None) -> float:
        """
        Compute weighted similarity between two feature sets.
        
        Args:
            features1: First feature set (from FeatureExtractionService.extract())
            features2: Second feature set (from FeatureExtractionService.extract())
            selected_categories: List of categories to use. If None, uses all available.
        
        Returns:
            Weighted similarity score in [0, 1]
        """
        if selected_categories is None:
            # Use all categories present in both feature sets
            selected_categories = list(set(features1.keys()) & set(features2.keys()))
        
        if not selected_categories:
            raise ValueError("No common categories found between feature sets")
        
        category_similarities = {}
        
        # Level 1: Compute similarity within each category
        for category in selected_categories:
            if category not in features1 or category not in features2:
                # Skip if category not present in both
                continue
            
            if category not in self.weights:
                # No weights defined for this category, skip
                continue
            
            cat_features1 = features1[category]
            cat_features2 = features2[category]
            extractor_weights = self.weights[category]
            
            # Compute weighted similarity within category
            category_sim = 0.0
            total_weight = 0.0
            
            for extractor_name, weight in extractor_weights.items():
                if extractor_name in cat_features1 and extractor_name in cat_features2:
                    vec1 = cat_features1[extractor_name]['vector']
                    vec2 = cat_features2[extractor_name]['vector']
                    
                    sim = self._cosine_similarity(vec1, vec2)
                    category_sim += weight * sim
                    total_weight += weight
            
            # Renormalize if some extractors were missing
            if total_weight > 0:
                category_sim = category_sim / total_weight
            
            category_similarities[category] = category_sim
        
        # Level 2: Combine category similarities with category weights
        if 'category_weights' not in self.weights:
            # No category weights, use equal weights
            return np.mean(list(category_similarities.values()))
        
        final_similarity = 0.0
        total_cat_weight = 0.0
        
        for category, sim in category_similarities.items():
            if category in self.weights['category_weights']:
                weight = self.weights['category_weights'][category]
                final_similarity += weight * sim
                total_cat_weight += weight
        
        # Renormalize if some categories were missing
        if total_cat_weight > 0:
            final_similarity = final_similarity / total_cat_weight
        else:
            # Fallback to mean if no weights available
            final_similarity = np.mean(list(category_similarities.values()))
        
        return float(final_similarity)
    
    def compute_category_similarity(self, features1: Dict[str, Any], 
                                   features2: Dict[str, Any],
                                   category: str) -> float:
        """
        Compute similarity for a specific category only.
        
        Args:
            features1: First feature set
            features2: Second feature set
            category: Category to compute similarity for
        
        Returns:
            Category-specific similarity score
        """
        if category not in features1 or category not in features2:
            return 0.0
        
        if category not in self.weights:
            return 0.0
        
        cat_features1 = features1[category]
        cat_features2 = features2[category]
        extractor_weights = self.weights[category]
        
        category_sim = 0.0
        total_weight = 0.0
        
        for extractor_name, weight in extractor_weights.items():
            if extractor_name in cat_features1 and extractor_name in cat_features2:
                vec1 = cat_features1[extractor_name]['vector']
                vec2 = cat_features2[extractor_name]['vector']
                
                sim = self._cosine_similarity(vec1, vec2)
                category_sim += weight * sim
                total_weight += weight
        
        if total_weight > 0:
            return float(category_sim / total_weight)
        return 0.0

