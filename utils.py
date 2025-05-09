import cv2  # For image loading and processing
def load_and_process_image(image_path):
    """
    Charge et prétraite une image pour l'analyse
    
    Args:
        image_path: Chemin vers l'image à charger
        
    Returns:
        Image prétraitée
    """
    # Charger l'image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionner si l'image est trop grande
    max_dim = 1024
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        if h > w:
            new_h, new_w = max_dim, int(w * max_dim / h)
        else:
            new_h, new_w = int(h * max_dim / w), max_dim
        image = cv2.resize(image, (new_w, new_h))
    
    return image