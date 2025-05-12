import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import joblib
import os
import traceback
from PIL import Image
import io
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import moments, moments_hu

# Importer les modules personnalisés
from morphology import (
    apply_morphological_operation, 
    calculate_morphological_laplacian, 
    calculate_gradient_morphology
)
from segmentation import segment_image_morphology
from features import extract_texture_features
from classification import create_segmentation_map, classify_pixels_directly
from visualization import (
    visualize_morphological_operations, 
    visualize_gradient_types, 
    compare_segmentation_methods
)
from utils import load_and_process_image

def main():
    st.set_page_config(page_title="Analyse d'images satellitaires", page_icon="🛰️", layout="wide")
    
    st.title("Analyse de textures et segmentation d'images satellitaires")
    st.write("""
    Cette application permet d'analyser des images satellitaires en utilisant des techniques de morphologie mathématique,
    de segmentation et de classification basée sur les textures.
    """)
    
    # Barre latérale
    st.sidebar.title("Options")
    
    # Section de chargement d'image
    st.sidebar.header("Chargement d'image")
    upload_option = st.sidebar.radio(
        "Source de l'image",
        ["Charger une image", "Utiliser une image d'exemple"]
    )
    
    # Charger le modèle
    model_path = 'models/terrain_classifier_model.pkl'
    
    # Vérifier si le dossier models existe, sinon le créer
    if not os.path.exists('models'):
        os.makedirs('models')
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.sidebar.success("Modèle de classification chargé")
        
        # Définir les classes disponibles
        class_names = ['agriculture', 'cloud', 'desert', 'forest', 'water']  # Ajustez selon vos classes
    else:
        st.sidebar.warning("Modèle non trouvé. Certaines fonctionnalités seront limitées.")
        model = None
        class_names = ['agriculture', 'cloud', 'desert', 'forest', 'water']  # Classes par défaut
    
    # Charger l'image
    if upload_option == "Charger une image":
        uploaded_file = st.sidebar.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = np.array(Image.open(uploaded_file))
            # Convertir en RGB si nécessaire
            if len(image.shape) > 2 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            st.info("Veuillez charger une image ou sélectionner une image d'exemple.")
            return
    else:
        # Images d'exemple
        example_images = {
            "Image agricole": "examples/agriculture.jpg",
            "Image forêt": "examples/forest.jpg",
            "Image désert": "examples/desert.jpg",
            "Image eau": "examples/water.jpg"
        }
        
        # Vérifier si le dossier examples existe, sinon le créer
        if not os.path.exists('examples'):
            os.makedirs('examples')
            st.warning("Dossier d'exemples créé. Veuillez y ajouter des images d'exemple.")
            st.write("Veuillez ajouter des images dans le dossier 'examples' puis redémarrer l'application.")
            return
            
        selected_example = st.sidebar.selectbox("Choisir une image d'exemple", list(example_images.keys()))
        image_path = example_images[selected_example]
        
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            st.error(f"Image d'exemple non trouvée: {image_path}")
            return
    
    # Afficher l'image originale
    st.subheader("Image originale")
    st.image(image, use_container_width=True)
    
    # Section de traitement morphologique
    st.sidebar.header("Traitement morphologique")
    
    operation = st.sidebar.selectbox(
        "Opération morphologique",
        ["Erosion", "Dilation", "Ouverture", "Fermeture", "Gradient", "Top-hat", "Black-hat", "Laplacien"]
    )
    
    kernel_size = st.sidebar.slider("Taille du noyau", 3, 15, 5, step=2)
    
    # Onglets pour les différentes analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Opérations morphologiques", 
        "Segmentation", 
        "Analyse de texture", 
        "Classification"
    ])
    
    with tab1:
        st.header("Opérations morphologiques")
        
        if st.button("Appliquer l'opération morphologique"):
            with st.spinner(f"Application de l'opération {operation}..."):
                try:
                    # Appliquer l'opération morphologique
                    if operation.lower() == "laplacien":
                        processed = calculate_morphological_laplacian(image, kernel_size)
                    else:
                        processed = apply_morphological_operation(image, operation.lower(), kernel_size)
                    
                    # Afficher l'image traitée
                    st.subheader(f"Résultat: {operation} (noyau {kernel_size}x{kernel_size})")
                    st.image(processed, use_container_width=True)
                    
                    # Afficher les histogrammes
                    if len(image.shape) > 2:
                        gray_original = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    else:
                        gray_original = image.copy()
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    axes[0].hist(gray_original.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
                    axes[0].set_title("Histogramme original")
                    axes[0].set_xlabel("Intensité")
                    axes[0].set_ylabel("Fréquence")
                    
                    axes[1].hist(processed.ravel(), bins=256, range=(0, 256), color='red', alpha=0.7)
                    axes[1].set_title(f"Histogramme après {operation}")
                    axes[1].set_xlabel("Intensité")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Erreur lors de l'application de l'opération morphologique: {e}")
                    st.code(traceback.format_exc())
    
    with tab2:
        st.header("Segmentation d'image")
        
        segmentation_method = st.selectbox(
            "Méthode de segmentation",
            ["Watershed", "Region Growing"]
        )
        
        if st.button("Segmenter l'image"):
            with st.spinner(f"Segmentation en cours avec la méthode {segmentation_method}..."):
                try:
                    # Effectuer la segmentation
                    labels, n_segments = segment_image_morphology(image, method=segmentation_method.lower().replace(" ", "_"))
                    
                    st.subheader(f"Résultat de segmentation: {n_segments} segments identifiés")
                    
                    # Créer une version colorée des labels pour visualisation
                    if len(image.shape) == 2:
                        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    else:
                        image_color = image.copy()
                    
                    # Générer des couleurs aléatoires pour chaque label
                    np.random.seed(42)  # Pour la reproductibilité
                    colors = np.random.randint(0, 255, size=(n_segments + 2, 3), dtype=np.uint8)
                    colors[0] = [0, 0, 0]  # Fond en noir
                    
                    # Créer une image segmentée en couleur
                    segmented_img = np.zeros_like(image_color)
                    for i in range(1, n_segments + 2):
                        mask = (labels == i)
                        if np.any(mask):
                            for c in range(3):
                                segmented_img[:,:,c][mask] = colors[i][c]
                    
                    # Créer un blending de l'image originale et des segments
                    alpha = 0.7
                    blended = cv2.addWeighted(
                        image_color, 1 - alpha, 
                        segmented_img.astype(np.uint8), alpha, 
                        0
                    )
                    
                    # Afficher les résultats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(segmented_img, caption="Segments identifiés", use_container_width=True)
                    with col2:
                        st.image(blended, caption="Superposition", use_container_width=True)
                    
                    # Comparaison des méthodes
                    st.subheader("Comparaison des méthodes de segmentation")
                    fig = compare_segmentation_methods(image)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Erreur lors de la segmentation: {e}")
                    st.code(traceback.format_exc())
    
    with tab3:
        st.header("Analyse de texture")
        
        if st.button("Analyser les textures"):
            with st.spinner("Analyse des textures en cours..."):
                try:
                    # Si l'image est en couleur, la convertir en niveaux de gris pour l'analyse
                    if len(image.shape) > 2:
                        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = image.copy()
                    
                    # Redimensionner pour standardiser
                    resized = cv2.resize(gray, (128, 128))
                    
                    # Calculer les matrices GLCM
                    distances = [1, 2, 3]
                    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
                    glcm = graycomatrix(resized, distances, angles, 256, symmetric=True, normed=True)
                    
                    # Calculer les propriétés GLCM
                    contrast = graycoprops(glcm, 'contrast')
                    dissimilarity = graycoprops(glcm, 'dissimilarity')
                    homogeneity = graycoprops(glcm, 'homogeneity')
                    energy = graycoprops(glcm, 'energy')
                    correlation = graycoprops(glcm, 'correlation')
                    
                    # Calculer LBP
                    lbp = local_binary_pattern(resized, P=8, R=1, method='uniform')
                    
                    # Calculer les moments d'Hu (alternative à Haralick)
                    m = moments(resized)
                    hu = moments_hu(m)
                    
                    # Afficher les résultats
                    st.subheader("Caractéristiques de texture")
                    
                    # Affichage des images
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(gray, caption="Image en niveaux de gris", use_container_width=True)
                    with col2:
                        st.image(lbp, caption="Local Binary Pattern", use_container_width=True, clamp=True)
                    
                    # Histogramme LBP
                    hist_lbp, _ = np.histogram(lbp, bins=10, range=(0, 10))
                    hist_lbp = hist_lbp.astype("float") / (hist_lbp.sum() + 1e-6)
                    
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    
                    # Histogramme LBP
                    axes[0, 0].bar(range(10), hist_lbp)
                    axes[0, 0].set_title('Histogramme LBP')
                    axes[0, 0].set_xlabel('Bins')
                    axes[0, 0].set_ylabel('Fréquence normalisée')
                    
                    # Propriétés GLCM en fonction de la distance
                    x_dist = distances
                    props = {'contrast': contrast, 'homogeneity': homogeneity, 'energy': energy, 'correlation': correlation}
                    for prop_name, prop_values in props.items():
                        for j, angle in enumerate(angles):
                            angle_deg = int(angle * 180 / np.pi)
                            axes[0, 1].plot(x_dist, prop_values[:, j], marker='o', label=f"{prop_name} ({angle_deg}°)")
                    
                    axes[0, 1].set_title('Propriétés GLCM vs Distance')
                    axes[0, 1].set_xlabel('Distance')
                    axes[0, 1].set_ylabel('Valeur')
                    axes[0, 1].legend()
                    
                    # Gradient morphologique
                    gradient = apply_morphological_operation(gray, 'gradient', 3)
                    axes[1, 0].imshow(gradient, cmap='gray')
                    axes[1, 0].set_title('Gradient morphologique')
                    axes[1, 0].axis('off')
                    
                    # Histogramme du gradient
                    axes[1, 1].hist(gradient.ravel(), bins=50, color='green', alpha=0.7)
                    axes[1, 1].set_title("Histogramme du gradient")
                    axes[1, 1].set_xlabel("Intensité")
                    axes[1, 1].set_ylabel("Fréquence")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Statistiques de texture
                    st.subheader("Statistiques de texture")
                    stats_col1, stats_col2 = st.columns(2)
                    
                    with stats_col1:
                        st.write("**Statistiques globales:**")
                        st.write(f"- Moyenne: {np.mean(gray):.2f}")
                        st.write(f"- Écart-type: {np.std(gray):.2f}")
                        st.write(f"- Contraste moyen (GLCM): {np.mean(contrast):.4f}")
                        st.write(f"- Homogénéité moyenne (GLCM): {np.mean(homogeneity):.4f}")
                    
                    with stats_col2:
                        st.write("**Statistiques du gradient:**")
                        st.write(f"- Moyenne du gradient: {np.mean(gradient):.2f}")
                        st.write(f"- Écart-type du gradient: {np.std(gradient):.2f}")
                        st.write(f"- Maximum du gradient: {np.max(gradient):.2f}")
                        st.write(f"- Minimum du gradient: {np.min(gradient):.2f}")
                    
                    # Moments d'Hu
                    st.subheader("Moments d'Hu (invariants de forme)")
                    st.write("Ces moments sont invariants par translation, rotation et échelle")
                    
                    hu_data = pd.DataFrame({
                        "Moment": [f"h{i+1}" for i in range(len(hu))],
                        "Valeur": ["{:.2e}".format(h) for h in hu]
                    })
                    st.table(hu_data)
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse de texture: {e}")
                    st.code(traceback.format_exc())
    
    with tab4:
        st.header("Classification")
        
        if model is None:
            st.warning("Aucun modèle de classification n'a été chargé. Veuillez d'abord entrainer un modèle.")
        else:
            classification_method = st.radio(
                "Méthode de classification",
                ["Classification par segments", "Classification directe par pixels"]
            )
            
            if st.button("Classifier l'image"):
                with st.spinner("Classification en cours..."):
                    try:
                        if classification_method == "Classification par segments":
                            # Méthode originale
                            segmented, segment_classes, blended = create_segmentation_map(image, model, class_names)
                        else:
                            # Nouvelle méthode plus simple
                            grid_size = st.slider("Taille de la grille (pixels)", 8, 32, 16)
                            segmented, segment_classes, blended = classify_pixels_directly(image, model, class_names, grid_size)
                        
                        st.subheader("Résultats de classification")
                        
                        # Afficher les images
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(segmented, caption="Classification des segments", use_container_width=True)
                        with col2:
                            st.image(blended, caption="Superposition", use_container_width=True)
                        
                        # Afficher la distribution des classes
                        st.subheader("Distribution des types de terrain")
                        
                        # Compter les occurrences de chaque classe
                        class_counts = {}
                        for class_name in segment_classes.values():
                            if class_name in class_counts:
                                class_counts[class_name] += 1
                            else:
                                class_counts[class_name] = 1
                        
                        # Vérifier si des classes ont été identifiées
                        if len(class_counts) > 0 and not ('placeholder' in class_counts and len(class_counts) == 1):
                            # Créer un graphique circulaire
                            fig, ax = plt.subplots(figsize=(8, 8))
                            ax.pie(
                                list(class_counts.values()), 
                                labels=list(class_counts.keys()), 
                                autopct='%1.1f%%',
                                colors=plt.cm.tab10.colors[:len(class_counts)]
                            )
                            ax.set_title('Distribution des types de terrain')
                            st.pyplot(fig)
                            
                            # Afficher les statistiques sous forme de tableau
                            st.subheader("Statistiques des classes identifiées")
                            
                            # Créer un DataFrame pour les statistiques
                            stats_data = []
                            total_segments = sum(class_counts.values())
                            
                            for class_name, count in class_counts.items():
                                percentage = count / total_segments * 100
                                stats_data.append({
                                    "Classe": class_name,
                                    "Nombre de segments": count,
                                    "Pourcentage": f"{percentage:.1f}%"
                                })
                            
                            stats_df = pd.DataFrame(stats_data)
                            st.table(stats_df)
                        else:
                            st.warning("Aucune classe n'a été identifiée. Essayez d'ajuster les paramètres ou d'utiliser une autre méthode de classification.")
                    
                    except Exception as e:
                        st.error(f"Erreur lors de la classification: {e}")
                        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()