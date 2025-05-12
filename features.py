import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import moments, moments_hu
from morphology import apply_morphological_operation

def extract_texture_features(image):
    """
    Version robuste de l'extraction de caractéristiques texturales
    avec gestion des valeurs manquantes et longueur constante
    """
    try:
        # Si l'image est en couleur, la convertir en niveaux de gris
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Vérifier si l'image contient des pixels valides
        if np.sum(gray) == 0 or gray.size == 0:
            # Retourner un vecteur de caractéristiques avec des valeurs par défaut
            return np.zeros(68)  # Longueur fixe du vecteur
        
        # Redimensionner l'image pour normaliser
        resized = cv2.resize(gray, (128, 128))
        
        # Caractéristiques GLCM
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Assurer la plage de valeurs correcte pour GLCM
        # Normaliser entre 0 et 255 et convertir en entier
        resized_norm = np.round((resized - np.min(resized)) * (255.0 / (np.max(resized) - np.min(resized) + 1e-10))).astype(np.uint8)
        
        glcm = graycomatrix(resized_norm, distances, angles, 256, symmetric=True, normed=True)
        
        # Extraire les propriétés GLCM avec gestion des erreurs
        try:
            contrast = graycoprops(glcm, 'contrast').flatten()
            dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
            homogeneity = graycoprops(glcm, 'homogeneity').flatten()
            energy = graycoprops(glcm, 'energy').flatten()
            correlation = graycoprops(glcm, 'correlation').flatten()
        except:
            # En cas d'erreur, utiliser des valeurs par défaut
            feature_length = len(distances) * len(angles)
            contrast = np.zeros(feature_length)
            dissimilarity = np.zeros(feature_length)
            homogeneity = np.zeros(feature_length)
            energy = np.zeros(feature_length)
            correlation = np.zeros(feature_length)
        
        # Caractéristiques statistiques de base
        mean = np.mean(resized)
        std = np.std(resized)
        
        # Ajouter des caractéristiques statistiques supplémentaires
        variance = np.var(resized)
        skewness = np.mean(((resized - mean)/(std + 1e-10))**3) if std > 0 else 0
        kurtosis = np.mean(((resized - mean)/(std + 1e-10))**4) if std > 0 else 0
        
        # Caractéristiques de gradient
        try:
            gradient = apply_morphological_operation(resized, 'gradient', 3)
            mean_gradient = np.mean(gradient)
            std_gradient = np.std(gradient)
        except:
            mean_gradient = 0
            std_gradient = 0
        
        # Moments d'Hu avec gestion des erreurs
        try:
            m = moments(resized)
            hu_moments = moments_hu(m)
        except:
            hu_moments = np.zeros(7)  # 7 moments d'Hu
        
        # Caractéristiques LBP (Local Binary Pattern)
        try:
            lbp = local_binary_pattern(resized, P=8, R=1, method='uniform')
            hist_lbp, _ = np.histogram(lbp, bins=10, range=(0, 10))
            hist_lbp = hist_lbp.astype("float") / (hist_lbp.sum() + 1e-10)
        except:
            hist_lbp = np.zeros(10)
        
        # Histogramme d'intensité
        try:
            hist_intensity, _ = np.histogram(resized, bins=5, range=(0, 256))
            hist_intensity = hist_intensity.astype("float") / (hist_intensity.sum() + 1e-10)
        except:
            hist_intensity = np.zeros(5)
        
        # Concatenation de toutes les caractéristiques
        features = np.concatenate([
            contrast, dissimilarity, homogeneity, energy, correlation,
            [mean, std, variance, skewness, kurtosis, mean_gradient, std_gradient],
            hu_moments,
            hist_lbp,
            hist_intensity
        ])
        
        # Remplacer les valeurs NaN ou inf par zéro
        features = np.nan_to_num(features)
        
        return features
    
    except Exception as e:
        print(f"Erreur dans l'extraction des caractéristiques: {e}")
        # En cas d'erreur, retourner un vecteur de zéros
        return np.zeros(68)  # Longueur fixe du vecteur