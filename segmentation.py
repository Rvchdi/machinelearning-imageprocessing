import numpy as np
import cv2  # Pour le traitement d'image
import matplotlib.pyplot as plt  # Pour la visualisation    

def segment_image_morphology(image, method='watershed'):
    """
    Segmente l'image en utilisant des techniques morphologiques
    
    Args:
        image: Image d'entrée
        method: Méthode de segmentation ('watershed', 'region_growing')
    
    Returns:
        Image segmentée et labels des segments
    """
    # Conversion en niveaux de gris si nécessaire
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convertir en BGR pour watershed
    
    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if method.lower() == 'watershed':
        # Segmentation watershed
        # Calcul du gradient pour trouver les bordures
        gradient = apply_morphological_operation(blurred, 'gradient', 3)
        
        # Binarisation du gradient
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Transformation de distance
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        
        # Binarisation de la transformation de distance
        _, dist_bin = cv2.threshold(dist, 0.5*dist.max(), 255, cv2.THRESH_BINARY)
        dist_bin = dist_bin.astype(np.uint8)
        
        # Recherche des marqueurs (noyaux des régions)
        contours, _ = cv2.findContours(dist_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Créer une matrice de marqueurs avec le type correct (CV_32SC1)
        markers = np.zeros(gray.shape, dtype=np.int32)
        
        # Dessiner les contours des régions
        for i, contour in enumerate(contours):
            cv2.drawContours(markers, [contour], -1, i+1, -1)
        
        # Préparer l'image pour watershed (doit être CV_8UC3)
        if len(image.shape) < 3 or image.shape[2] != 3:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()
        
        # Appliquer watershed
        cv2.watershed(image_color, markers)
        
        return markers, len(contours)
    
    elif method.lower() == 'region_growing':
        # Implémentation simplifiée de croissance de région
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Trouver les composantes connectées
        num_labels, labels = cv2.connectedComponents(binary)
        
        return labels, num_labels - 1
    
    else:
        # Méthode par défaut: simple seuillage adaptatif
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        num_labels, labels = cv2.connectedComponents(binary)
        
        return labels, num_labels - 1