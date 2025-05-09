import numpy as np
import cv2  # For image processing
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern 
def extract_texture_features(image):
    """
    Extrait les caractéristiques de texture d'une image
    
    Args:
        image: Image en niveaux de gris
        
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
    
    # Caractéristiques de Haralick avec Mahotas
    haralick = mh.features.haralick(resized).mean(axis=0)
    
    # Caractéristiques LBP (Local Binary Pattern)
    lbp = local_binary_pattern(resized, P=8, R=1, method='uniform')
    hist_lbp, _ = np.histogram(lbp, bins=10, range=(0, 10))
    hist_lbp = hist_lbp.astype("float") / (hist_lbp.sum() + 1e-6)
    
    # Caractéristiques morphologiques
    # Gradient
    gradient = apply_morphological_operation(resized, 'gradient', 3)
    mean_gradient = np.mean(gradient)
    std_gradient = np.std(gradient)
    
    # Concatenation de toutes les caractéristiques
    features = np.concatenate([
        contrast, dissimilarity, homogeneity, energy, correlation,
        [mean, std, mean_gradient, std_gradient],
        haralick,
        hist_lbp
    ])
    
    return features