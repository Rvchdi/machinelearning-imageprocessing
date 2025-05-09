import numpy as np
import cv2  # Pour le traitement d'image
import matplotlib.pyplot as plt  # Pour la visualisation    
from segmentation import segment_image_morphology
from features import extract_texture_features

def create_segmentation_map(image, model, class_names):
    """
    Crée une carte de segmentation avec classification des régions
    
    Args:
        image: Image à analyser
        model: Modèle de classification
        class_names: Noms des classes
    """
    # Segmentation de l'image
    labels, n_segments = segment_image_morphology(image, method='watershed')
    
    # Classification de chaque segment
    segment_classes = {}
    
    # Image segmentée pour la visualisation
    segmented = np.zeros_like(image)
    
    # Si l'image est en niveaux de gris, la convertir en RGB
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_color = image.copy()
    
    # Couleurs pour chaque classe (à adapter selon vos classes spécifiques)
    colors = {}
    for i, name in enumerate(class_names):
        hue = i * 180 // len(class_names)
        # Convertir HSV en RGB pour générer des couleurs distinctes
        rgb = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
        colors[name] = rgb.tolist()
    
    # Analyser chaque segment
    for segment_id in range(1, n_segments + 2):
        # Créer un masque pour le segment actuel
        mask = (labels == segment_id)
        if not np.any(mask):
            continue
            
        # Extraire la région correspondante de l'image originale
        segment_img = image.copy()
        if len(image.shape) == 3:
            for c in range(3):
                segment_img[:,:,c] = segment_img[:,:,c] * mask
        else:
            segment_img = segment_img * mask
        
        # Calculer les caractéristiques de texture
        try:
            features = extract_texture_features(segment_img)
            
            # Prédire la classe
            predicted_class_idx = model.predict([features])[0]
            predicted_class = class_names[predicted_class_idx]
            
            # Stocker la classe prédite
            segment_classes[segment_id] = predicted_class
            
            # Colorier le segment
            if predicted_class in colors:
                if len(image.shape) == 3:
                    for c in range(3):
                        segmented[:,:,c][mask] = colors[predicted_class][c]
                else:
                    segmented[mask] = colors[predicted_class][0]  # Utiliser seulement le canal rouge
        except Exception as e:
            print(f"Erreur lors de la classification du segment {segment_id}: {e}")
    
    # Créer une version blended (superposition semi-transparente)
    alpha = 0.7
    blended = cv2.addWeighted(image_color, 1 - alpha, segmented.astype(np.uint8), alpha, 0)
    
    return segmented, segment_classes, blended