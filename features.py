import numpy as np
import cv2  # For image processing
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import moments, moments_hu
from morphology import apply_morphological_operation

def extract_texture_features_no_mahotas(image):
    """
    Extrait les caractéristiques de texture d'une image sans utiliser mahotas
    
    Args:
        image: Image en niveaux de gris ou en couleur
        
    Returns:
        Vecteur de caractéristiques
    """
    # Si l'image est en couleur, la convertir en niveaux de gris
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Redimensionner l'image pour normaliser
    resized = cv2.resize(gray, (128, 128))
    
    # Caractéristiques GLCM
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(resized, distances, angles, 256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    
    # Caractéristiques statistiques
    mean = np.mean(resized)
    std = np.std(resized)
    
    # Caractéristiques morphologiques - Gradient
    gradient = apply_morphological_operation(resized, 'gradient', 3)
    mean_gradient = np.mean(gradient)
    std_gradient = np.std(gradient)
    
    # Moments d'Hu (invariants par rotation, échelle et translation)
    m = moments(resized)
    hu_moments = moments_hu(m)
    
    # Caractéristiques LBP (Local Binary Pattern)
    lbp = local_binary_pattern(resized, P=8, R=1, method='uniform')
    hist_lbp, _ = np.histogram(lbp, bins=10, range=(0, 10))
    hist_lbp = hist_lbp.astype("float") / (hist_lbp.sum() + 1e-6)
    
    # Histogramme d'intensité (5 bins)
    hist_intensity, _ = np.histogram(resized, bins=5, range=(0, 256))
    hist_intensity = hist_intensity.astype("float") / (hist_intensity.sum() + 1e-6)
    
    # Caractéristiques supplémentaires
    variance = np.var(resized)
    skewness = np.mean(((resized - mean)/std)**3) if std > 0 else 0
    kurtosis = np.mean(((resized - mean)/std)**4) if std > 0 else 0
    
    # Concatenation de toutes les caractéristiques
    features = np.concatenate([
        contrast, dissimilarity, homogeneity, energy, correlation,
        [mean, std, variance, skewness, kurtosis, mean_gradient, std_gradient],
        hu_moments,
        hist_lbp,
        hist_intensity
    ])
    
    return features

# Garder la fonction originale pour compatibilité, mais utiliser la nouvelle implémentation
def extract_texture_features(image):
    """
    Version compatible de la fonction d'extraction de caractéristiques
    """
    return extract_texture_features_no_mahotas(image)