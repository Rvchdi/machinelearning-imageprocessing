import os  # For directory and file operations
import glob  # For file pattern matching
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
from skimage import io  # For image reading
import cv2  # For image processing

def visualize_sample_images(data_dir, n_samples=5):
    """
    Visualise quelques exemples d'images pour chaque classe
    """
    classes = os.listdir(data_dir)
    classes = [c for c in classes if os.path.isdir(os.path.join(data_dir, c))]
    
    fig, axes = plt.subplots(len(classes), n_samples, figsize=(15, 3*len(classes)))
    
    for i, class_name in enumerate(classes):
        class_samples = glob.glob(os.path.join(data_dir, class_name, '*.jpg'))
        selected_samples = np.random.choice(class_samples, min(n_samples, len(class_samples)), replace=False)
        
        for j, sample in enumerate(selected_samples):
            img = io.imread(sample)
            axes[i, j].imshow(img)
            axes[i, j].set_title(class_name)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_morphological_operations(image, kernel_size=5):
    """
    Visualise les résultats de différentes opérations morphologiques
    """
    operations = ['erosion', 'dilation', 'ouverture', 'fermeture', 'gradient', 'top-hat', 'black-hat']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Image originale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        axes[0].imshow(image)
    else:
        gray = image.copy()
        axes[0].imshow(gray, cmap='gray')
    
    axes[0].set_title('Image originale')
    axes[0].axis('off')
    
    # Appliquer et afficher les opérations
    for i, operation in enumerate(operations):
        result = apply_morphological_operation(image, operation, kernel_size)
        axes[i+1].imshow(result, cmap='gray')
        axes[i+1].set_title(f'{operation.capitalize()}, kernel={kernel_size}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_segmentation(image, method='watershed'):
    """
    Visualise le résultat de la segmentation
    """
    # Segmenter l'image
    labels, n_segments = segment_image_morphology(image, method=method)
    
    # Créer une image segmentée en couleur
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_color = image.copy()
    
    # Créer une version colorée des labels pour visualisation
    label_viz = np.zeros_like(image_color)
    
    # Générer des couleurs aléatoires pour chaque label
    np.random.seed(42)  # Pour la reproductibilité
    colors = np.random.randint(0, 255, size=(n_segments + 2, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Fond en noir
    
    # Colorier les segments
    for i in range(1, n_segments + 2):
        label_viz[labels == i] = colors[i]
    
    # Afficher
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image_color)
    axes[0].set_title('Image originale')
    axes[0].axis('off')
    
    axes[1].imshow(labels, cmap='jet')
    axes[1].set_title(f'Labels ({n_segments} segments)')
    axes[1].axis('off')
    
    axes[2].imshow(label_viz)
    axes[2].set_title('Segmentation colorée')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return labels, n_segments

def visualize_gradient_types(image, kernel_size=3):
    """
    Visualise les différents types de gradients morphologiques
    """
    # Calculer les gradients
    internal, external, morphological = calculate_gradient_morphology(image, kernel_size)
    
    # Afficher
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        axes[0].imshow(image)
    else:
        gray = image.copy()
        axes[0].imshow(gray, cmap='gray')
    
    axes[0].set_title('Image originale')
    axes[0].axis('off')
    
    axes[1].imshow(internal, cmap='gray')
    axes[1].set_title('Gradient interne')
    axes[1].axis('off')
    
    axes[2].imshow(external, cmap='gray')
    axes[2].set_title('Gradient externe')
    axes[2].axis('off')
    
    axes[3].imshow(morphological, cmap='gray')
    axes[3].set_title('Gradient morphologique')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
def compare_segmentation_methods(image):
    """
    Compare différentes méthodes de segmentation
    
    Args:
        image: Image à segmenter
    """
    methods = ['watershed', 'region_growing']
    
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(5 * (len(methods) + 1), 5))
    
    # Image originale
    if len(image.shape) == 2:
        axes[0].imshow(image, cmap='gray')
    else:
        axes[0].imshow(image)
    axes[0].set_title('Image originale')
    axes[0].axis('off')
    
    # Résultats des différentes méthodes
    for i, method in enumerate(methods):
        labels, n_segments = segment_image_morphology(image, method=method)
        
        # Afficher les labels en couleurs
        axes[i + 1].imshow(labels, cmap='jet')
        axes[i + 1].set_title(f'Méthode: {method}\n{n_segments} segments')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()