import cv2  # For image processing
import numpy as np  # For numerical operations
import os  # For file operations
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
        markers = np.zeros_like(gray)
        for i, contour in enumerate(contours):
            cv2.drawContours(markers, [contour], -1, i+1, -1)
        
        # Appliquer watershed
        if len(image.shape) == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()
        
        markers_copy = markers.copy()
        cv2.watershed(image_color, markers_copy)
        
        return markers_copy, len(contours)
    
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
def apply_morphological_operation(image, operation, kernel_size):
    """
    Applique une opération morphologique à l'image
    
    Args:
        image: Image d'entrée
        operation: Nom de l'opération ('erosion', 'dilation', etc.)
        kernel_size: Taille du noyau (élément structurant)
    
    Returns:
        Image après application de l'opération
    """
    # Conversion en niveaux de gris si l'image est en couleur
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Création du noyau
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Application de l'opération
    if operation.lower() == 'erosion':
        result = cv2.erode(gray, kernel, iterations=1)
    elif operation.lower() == 'dilation':
        result = cv2.dilate(gray, kernel, iterations=1)
    elif operation.lower() == 'ouverture':
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    elif operation.lower() == 'fermeture':
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    elif operation.lower() == 'gradient':
        result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    elif operation.lower() == 'top-hat':
        result = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    elif operation.lower() == 'black-hat':
        result = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    else:
        result = gray  # Opération non reconnue
    
    return result

def calculate_morphological_laplacian(image, kernel_size):
    """
    Calcule le laplacien morphologique: (dilation(dilation(img)) - 2*dilation(img) + img)
    """
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Première et seconde dilatation
    dilated = cv2.dilate(gray, kernel, iterations=1)
    double_dilated = cv2.dilate(dilated, kernel, iterations=1)
    
    # Laplacien morphologique
    laplacian = cv2.subtract(double_dilated, cv2.add(dilated, dilated))
    laplacian = cv2.add(laplacian, gray)
    
    return laplacian

def calculate_gradient_morphology(image, kernel_size=3):
    """
    Calcule les trois types de gradients morphologiques:
    - Gradient interne: différence entre l'image originale et son érosion
    - Gradient externe: différence entre la dilatation et l'image originale
    - Gradient morphologique symétrique: différence entre la dilatation et l'érosion
    """
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Érosion et dilatation
    erosion = cv2.erode(gray, kernel, iterations=1)
    dilation = cv2.dilate(gray, kernel, iterations=1)
    
    # Gradients
    internal_gradient = cv2.subtract(gray, erosion)
    external_gradient = cv2.subtract(dilation, gray)
    morphological_gradient = cv2.subtract(dilation, erosion)
    
    return internal_gradient, external_gradient, morphological_gradient

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
        markers = np.zeros_like(gray)
        for i, contour in enumerate(contours):
            cv2.drawContours(markers, [contour], -1, i+1, -1)
        
        # Appliquer watershed
        if len(image.shape) == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()
        
        markers_copy = markers.copy()
        cv2.watershed(image_color, markers_copy)
        
        return markers_copy, len(contours)
    
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