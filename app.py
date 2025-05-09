import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import joblib
import os
from PIL import Image
import io

# Importer vos modules personnalisés
from morphology import apply_morphological_operation, calculate_morphological_laplacian, calculate_gradient_morphology, segment_image_morphology
from segmentation import segment_image_morphology
from features import extract_texture_features
from classification import create_segmentation_map
from visualization import visualize_morphological_operations, visualize_gradient_types, compare_segmentation_methods
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
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.sidebar.success("Modèle de classification chargé")
        
        # Définir les classes disponibles
        class_names = ['agriculture', 'cloud', 'desert', 'forest', 'water']  # Ajustez selon vos classes
    else:
        st.sidebar.error("Modèle non trouvé. Veuillez d'abord entraîner un modèle.")
        model = None
        class_names = []
    
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
    
    with tab2:
        # Code pour l'onglet de segmentation
        pass
    
    with tab3:
        # Code pour l'onglet d'analyse de texture
        pass
    
    with tab4:
        # Code pour l'onglet de classification
        pass

if __name__ == "__main__":
    main()