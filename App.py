import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Input, Concatenate, UpSampling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import os
import pickle
import io
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, jaccard_score
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json

# Configuração da página
st.set_page_config(
    page_title="Análise Sísmica Avançada com CNN",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🛢️ Sistema Avançado de Análise de Imagens Sísmicas com CNN")
st.markdown("### Identificação de Padrões para Exploração de Hidrocarbonetos com Suporte a Masks")

class SeismicCNN:
    def __init__(self, input_shape=(128, 128, 3), num_classes=3, use_masks=False, use_transfer_learning=False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_masks = use_masks
        self.use_transfer_learning = use_transfer_learning
        self.model = None
        self.history = None
        self.data_generator = None
        
    def build_model(self):
        """Constrói a arquitetura da CNN com opções avançadas"""
        if self.use_transfer_learning:
            return self._build_transfer_learning_model()
        elif self.use_masks:
            return self._build_unet_model()
        else:
            return self._build_standard_cnn()
    
    def _build_standard_cnn(self):
        """CNN padrão melhorada"""
        model = tf.keras.Sequential([
            # Primeira camada convolucional
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Segunda camada convolucional
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Terceira camada convolucional
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Quarta camada convolucional
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Global Average Pooling
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Camadas densas
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def _build_transfer_learning_model(self):
        """Modelo com Transfer Learning usando ResNet50"""
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Congelar camadas iniciais
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def _build_unet_model(self):
        """Modelo U-Net para segmentação com masks"""
        inputs = Input(shape=self.input_shape)
        
        # Encoder
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        drop5 = Dropout(0.5)(conv5)
        
        # Decoder
        up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = Concatenate()([drop4, up6])
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
        
        up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = Concatenate()([conv3, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
        
        up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = Concatenate()([conv2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
        
        up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = Concatenate()([conv1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)
        
        # Saída para segmentação
        segmentation_output = Conv2D(self.num_classes, 1, activation='softmax', name='segmentation')(conv9)
        
        # Branch para classificação
        gap = tf.keras.layers.GlobalAveragePooling2D()(conv5)
        dense1 = Dense(512, activation='relu')(gap)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.5)(dense1)
        classification_output = Dense(self.num_classes, activation='softmax', name='classification')(dense1)
        
        model = Model(inputs=inputs, outputs=[segmentation_output, classification_output])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss={
                'segmentation': 'categorical_crossentropy',
                'classification': 'categorical_crossentropy'
            },
            loss_weights={'segmentation': 0.7, 'classification': 0.3},
            metrics={
                'segmentation': ['accuracy'],
                'classification': ['accuracy']
            }
        )
        
        self.model = model
        return model
    
    def setup_data_augmentation(self):
        """Configura data augmentation específico para dados sísmicos"""
        self.data_generator = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='reflect'
        )
    
    def preprocess_image(self, image, target_size=(128, 128)):
        """Pré-processa uma imagem"""
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif hasattr(image, 'read'):
            image.seek(0)
            file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(image)
            if len(img.shape) == 3 and img.shape[2] == 3:
                pass
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
        
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        return img
    
    def predict_single_image(self, image):
        """Faz predição para uma única imagem"""
        processed_img = self.preprocess_image(image)
        processed_img = np.expand_dims(processed_img, axis=0)
        
        if self.use_masks:
            predictions = self.model.predict(processed_img)
            return predictions[1]  # Retorna classificação
        else:
            prediction = self.model.predict(processed_img)
            return prediction
    
    def generate_gradcam(self, image, class_index, layer_name=None):
        """Gera Grad-CAM para explicabilidade"""
        processed_img = self.preprocess_image(image)
        processed_img = np.expand_dims(processed_img, axis=0)
        
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4:
                    layer_name = layer.name
                    break
        
        grad_model = tf.keras.models.Model(
            [self.model.inputs], 
            [self.model.get_layer(layer_name).output, self.model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(processed_img)
            loss = predictions[:, class_index]
        
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        
        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = gate_f * gate_r * grads
        
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
        
        cam = cv2.resize(cam.numpy(), (128, 128))
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        
        return cam

# Funções auxiliares - CORRIGIDAS PARA EVITAR PROBLEMAS DE DIMENSÃO
def load_images_and_labels(image_files, labels, target_size=(128, 128)):
    """Carrega apenas imagens - FUNÇÃO SEGURA"""
    images = []
    processed_labels = []
    valid_indices = []
    
    for i, (img_file, label) in enumerate(zip(image_files, labels)):
        try:
            img_file.seek(0)
            image = Image.open(img_file)
            image = image.convert('RGB')
            image = image.resize(target_size)
            
            img_array = np.array(image).astype(np.float32) / 255.0
            images.append(img_array)
            processed_labels.append(label)
            valid_indices.append(i)
            
        except Exception as e:
            st.error(f"Erro ao processar {img_file.name}: {str(e)}")
            continue
    
    return np.array(images), np.array(processed_labels), valid_indices

def load_images_masks_and_labels(image_files, mask_files, labels, target_size=(128, 128)):
    """Carrega imagens + masks - FUNÇÃO SEGURA COM VALIDAÇÃO"""
    if len(image_files) != len(mask_files) or len(image_files) != len(labels):
        raise ValueError(f"Dimensões incompatíveis: {len(image_files)} imagens, {len(mask_files)} masks, {len(labels)} labels")
    
    images = []
    masks = []
    processed_labels = []
    valid_indices = []
    
    for i, (img_file, mask_file, label) in enumerate(zip(image_files, mask_files, labels)):
        try:
            # Processar imagem
            img_file.seek(0)
            image = Image.open(img_file)
            image = image.convert('RGB')
            image = image.resize(target_size)
            img_array = np.array(image).astype(np.float32) / 255.0
            
            # Processar mask
            mask_file.seek(0)
            mask = Image.open(mask_file)
            mask = mask.convert('L')
            mask = mask.resize(target_size)
            mask_array = np.array(mask)
            
            # Normalizar mask para classes 0-2
            mask_array = (mask_array / 127.5).astype(np.uint8)
            mask_array = np.clip(mask_array, 0, 2)
            mask_categorical = tf.keras.utils.to_categorical(mask_array, 3)
            
            # Só adiciona se ambos foram processados com sucesso
            images.append(img_array)
            masks.append(mask_categorical)
            processed_labels.append(label)
            valid_indices.append(i)
            
        except Exception as e:
            st.error(f"Erro ao processar {img_file.name} ou {mask_file.name}: {str(e)}")
            continue
    
    if len(images) == 0:
        raise ValueError("Nenhuma imagem/mask foi processada com sucesso")
    
    return np.array(images), np.array(masks), np.array(processed_labels), valid_indices

def safe_metrics_calculation(y_true, y_pred, class_names):
    """Calcula métricas de forma segura, evitando erros de dimensão"""
    try:
        # Garantir que ambos são arrays numpy 1D
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Debug info
        st.write(f"**Debug Métricas:**")
        st.write(f"- y_true shape: {y_true.shape}, unique values: {np.unique(y_true)}")
        st.write(f"- y_pred shape: {y_pred.shape}, unique values: {np.unique(y_pred)}")
        
        # Verificar se têm o mesmo tamanho
        if len(y_true) != len(y_pred):
            st.error(f"Erro: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")
            return None, None, None, None
        
        # Calcular métricas
        accuracy = accuracy_score(y_true, y_pred)
        
        # Relatório de classificação com tratamento de erro
        try:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0, labels=[0, 1, 2])
            precision_avg = report['macro avg']['precision']
            recall_avg = report['macro avg']['recall']
        except Exception as e:
            st.warning(f"Erro no relatório de classificação: {str(e)}")
            precision_avg = 0.0
            recall_avg = 0.0
            report = None
        
        # Matriz de confusão
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        except Exception as e:
            st.warning(f"Erro na matriz de confusão: {str(e)}")
            cm = None
        
        return accuracy, precision_avg, recall_avg, cm, report
        
    except Exception as e:
        st.error(f"Erro geral no cálculo de métricas: {str(e)}")
        return None, None, None, None, None

def plot_training_history(history, use_masks=False):
    """Plota o histórico de treinamento"""
    try:
        if use_masks:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Acurácia - Classificação', 'Perda - Classificação', 
                              'Acurácia - Segmentação', 'Perda - Segmentação'),
            )
            
            epochs = range(1, len(history.history['classification_accuracy']) + 1)
            
            # Classificação
            fig.add_trace(go.Scatter(x=list(epochs), y=history.history['classification_accuracy'], 
                                    mode='lines', name='Treino', line=dict(color='blue')), row=1, col=1)
            if 'val_classification_accuracy' in history.history:
                fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_classification_accuracy'], 
                                        mode='lines', name='Validação', line=dict(color='red')), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=list(epochs), y=history.history['classification_loss'], 
                                    mode='lines', name='Treino', line=dict(color='blue')), row=1, col=2)
            if 'val_classification_loss' in history.history:
                fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_classification_loss'], 
                                        mode='lines', name='Validação', line=dict(color='red')), row=1, col=2)
            
            # Segmentação
            fig.add_trace(go.Scatter(x=list(epochs), y=history.history['segmentation_accuracy'], 
                                    mode='lines', name='Treino', line=dict(color='green')), row=2, col=1)
            if 'val_segmentation_accuracy' in history.history:
                fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_segmentation_accuracy'], 
                                        mode='lines', name='Validação', line=dict(color='orange')), row=2, col=1)
            
            fig.add_trace(go.Scatter(x=list(epochs), y=history.history['segmentation_loss'], 
                                    mode='lines', name='Treino', line=dict(color='green')), row=2, col=2)
            if 'val_segmentation_loss' in history.history:
                fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_segmentation_loss'], 
                                        mode='lines', name='Validação', line=dict(color='orange')), row=2, col=2)
        else:
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Acurácia', 'Perda'))
            
            epochs = range(1, len(history.history['accuracy']) + 1)
            
            fig.add_trace(go.Scatter(x=list(epochs), y=history.history['accuracy'], 
                                    mode='lines', name='Treino', line=dict(color='blue')), row=1, col=1)
            if 'val_accuracy' in history.history:
                fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_accuracy'], 
                                        mode='lines', name='Validação', line=dict(color='red')), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=list(epochs), y=history.history['loss'], 
                                    mode='lines', name='Treino', line=dict(color='blue')), row=1, col=2)
            if 'val_loss' in history.history:
                fig.add_trace(go.Scatter(x=list(epochs), y=history.history['val_loss'], 
                                        mode='lines', name='Validação', line=dict(color='red')), row=1, col=2)
        
        fig.update_layout(height=600, showlegend=False, title_text="Métricas de Treinamento")
        return fig
        
    except Exception as e:
        st.error(f"Erro ao plotar histórico de treinamento: {str(e)}")
        return None

def plot_confusion_matrix_safe(cm, class_names):
    """Plota matriz de confusão de forma segura"""
    try:
        if cm is None:
            st.warning("Matriz de confusão não disponível")
            return None
            
        fig = px.imshow(cm, 
                        labels=dict(x="Predito", y="Real", color="Contagem"),
                        x=class_names,
                        y=class_names,
                        color_continuous_scale='Blues',
                        text_auto=True)
        fig.update_layout(title="Matriz de Confusão")
        return fig
        
    except Exception as e:
        st.error(f"Erro ao criar matriz de confusão: {str(e)}")
        return None

def plot_gradcam_overlay(original_image, gradcam, alpha=0.6):
    """Cria sobreposição de Grad-CAM"""
    try:
        gradcam = (gradcam * 255).astype(np.uint8)
        gradcam = cv2.applyColorMap(gradcam, cv2.COLORMAP_JET)
        gradcam = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)
        
        if original_image.shape[:2] != gradcam.shape[:2]:
            original_image = cv2.resize(original_image, (gradcam.shape[1], gradcam.shape[0]))
        
        overlay = cv2.addWeighted(original_image, 1-alpha, gradcam, alpha, 0)
        return overlay
        
    except Exception as e:
        st.error(f"Erro ao criar Grad-CAM: {str(e)}")
        return original_image

# Interface Streamlit Principal
def main():
    # Sidebar
    st.sidebar.header("⚙️ Configurações")
    
    # Definição das classes
    class_names = ['Não Reservatório', 'Possível Reservatório', 'Provável Reservatório']
    
    # Configurações do modelo
    st.sidebar.subheader("🧠 Arquitetura do Modelo")
    use_masks = st.sidebar.checkbox("Usar Masks para Segmentação", value=False)
    use_transfer_learning = st.sidebar.checkbox("Transfer Learning (ResNet50)", value=False)
    use_data_augmentation = st.sidebar.checkbox("Data Augmentation", value=True)
    
    # Seção de upload de dados
    st.header("📁 Carregamento de Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dados de Treinamento")
        train_files = st.file_uploader(
            "Carregar imagens de treino",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="train"
        )
        
        # Upload de masks se habilitado - CORRIGIDO
        train_masks = None
        if use_masks:
            if train_files:
                st.info(f"🎯 Modo Segmentação: Carregue {len(train_files)} masks correspondentes")
                train_masks = st.file_uploader(
                    f"Masks de treino ({len(train_files)} necessárias)",
                    type=['png', 'jpg', 'jpeg'],
                    accept_multiple_files=True,
                    key="train_masks",
                    help="Cada mask deve corresponder a uma imagem na mesma ordem"
                )
                
                if train_masks:
                    if len(train_masks) == len(train_files):
                        st.success(f"✅ {len(train_masks)} masks válidas carregadas")
                        
                        # Mostrar correspondência
                        with st.expander("Ver correspondência Imagem-Mask"):
                            for i, (img, mask) in enumerate(zip(train_files, train_masks)):
                                st.write(f"{i+1}. {img.name} ↔ {mask.name}")
                                
                    else:
                        st.error(f"❌ Erro: {len(train_masks)} masks ≠ {len(train_files)} imagens")
                        st.info("💡 Carregue exatamente a mesma quantidade de masks e imagens")
                        train_masks = None
            else:
                st.info("📤 Primeiro carregue as imagens de treino")
        
        if train_files:
            st.success(f"✅ {len(train_files)} imagens carregadas")
            
            # Labels para dados de treino
            st.subheader("Rótulos das Imagens de Treino")
            train_labels = []
            for i, file in enumerate(train_files):
                label = st.selectbox(
                    f"Classe para {file.name}:",
                    options=[0, 1, 2],
                    format_func=lambda x: class_names[x],
                    key=f"train_label_{i}"
                )
                train_labels.append(label)
    
    with col2:
        st.subheader("Dados de Teste")
        test_files = st.file_uploader(
            "Carregar imagens de teste",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="test"
        )
        
        # Upload de masks de teste - CORRIGIDO
        test_masks = None
        if use_masks:
            if test_files:
                st.info(f"🎯 Modo Segmentação: Carregue {len(test_files)} masks correspondentes")
                test_masks = st.file_uploader(
                    f"Masks de teste ({len(test_files)} necessárias)",
                    type=['png', 'jpg', 'jpeg'],
                    accept_multiple_files=True,
                    key="test_masks",
                    help="Cada mask deve corresponder a uma imagem na mesma ordem"
                )
                
                if test_masks:
                    if len(test_masks) == len(test_files):
                        st.success(f"✅ {len(test_masks)} masks válidas carregadas")
                        
                        # Mostrar correspondência
                        with st.expander("Ver correspondência Imagem-Mask"):
                            for i, (img, mask) in enumerate(zip(test_files, test_masks)):
                                st.write(f"{i+1}. {img.name} ↔ {mask.name}")
                                
                    else:
                        st.error(f"❌ Erro: {len(test_masks)} masks ≠ {len(test_files)} imagens")
                        st.info("💡 Carregue exatamente a mesma quantidade de masks e imagens")
                        test_masks = None
            else:
                st.info("📤 Primeiro carregue as imagens de teste")
        
        if test_files:
            st.success(f"✅ {len(test_files)} imagens carregadas")
            
            # Labels para dados de teste
            st.subheader("Rótulos das Imagens de Teste")
            test_labels = []
            for i, file in enumerate(test_files):
                label = st.selectbox(
                    f"Classe para {file.name}:",
                    options=[0, 1, 2],
                    format_func=lambda x: class_names[x],
                    key=f"test_label_{i}"
                )
                test_labels.append(label)
    
    # Parâmetros de treinamento
    st.sidebar.subheader("Parâmetros de Treinamento")
    epochs = st.sidebar.slider("Número de Épocas", 10, 200, 50)
    batch_size = st.sidebar.slider("Batch Size", 4, 32, 16)
    
    # Status do modo de treinamento
    if use_masks:
        has_valid_train_masks = train_masks and len(train_masks) == len(train_files) if train_files else False
        if has_valid_train_masks:
            st.sidebar.success("✅ Modo: Segmentação + Classificação")
        else:
            st.sidebar.warning("⚠️ Aguardando masks válidas")
    else:
        st.sidebar.info("ℹ️ Modo: Classificação")
    
    # Botão de treinamento
    if st.button("🚀 Iniciar Treinamento", type="primary"):
        if not train_files or len(train_labels) != len(train_files):
            st.error("❌ Carregue as imagens e defina os rótulos!")
            return
        
        # Determinar modo de treinamento
        train_with_masks = use_masks and train_masks and len(train_masks) == len(train_files)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # CARREGAR DADOS
            status_text.text("Carregando dados...")
            progress_bar.progress(20)
            
            if train_with_masks:
                st.info("🎯 Iniciando treinamento com segmentação...")
                X_train, masks_train, y_train, valid_train_indices = load_images_masks_and_labels(
                    train_files, train_masks, train_labels
                )
                st.write(f"✅ Carregados: {len(X_train)} pares imagem-mask")
            else:
                st.info("🎯 Iniciando treinamento com classificação...")
                X_train, y_train, valid_train_indices = load_images_and_labels(train_files, train_labels)
                masks_train = None
                st.write(f"✅ Carregadas: {len(X_train)} imagens")
            
            # PREPARAR DADOS
            status_text.text("Preparando dados...")
            progress_bar.progress(40)
            
            y_train_cat = tf.keras.utils.to_categorical(y_train, 3)
            
            # DIVISÃO TREINO/VALIDAÇÃO
            if train_with_masks:
                # COM MASKS
                X_train_split, X_val_split, y_train_split, y_val_split, masks_train_split, masks_val_split = train_test_split(
                    X_train, y_train_cat, masks_train, test_size=0.2, random_state=42
                )
                st.write(f"✅ Divisão: {len(X_train_split)} treino, {len(X_val_split)} validação (com masks)")
            else:
                # SEM MASKS
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train_cat, test_size=0.2, random_state=42
                )
                st.write(f"✅ Divisão: {len(X_train_split)} treino, {len(X_val_split)} validação")
            
            # CRIAR MODELO
            status_text.text("Construindo modelo...")
            progress_bar.progress(60)
            
            cnn = SeismicCNN(
                use_masks=train_with_masks,
                use_transfer_learning=use_transfer_learning
            )
            model = cnn.build_model()
            
            st.write(f"✅ Modelo criado: {model.count_params():,} parâmetros")
            
            if use_data_augmentation:
                cnn.setup_data_augmentation()
                st.write("✅ Data augmentation configurado")
            
            # TREINAMENTO
            status_text.text("Treinando modelo...")
            progress_bar.progress(80)
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7)
            ]
            
            if train_with_masks:
                # Treinamento com masks
                history = model.fit(
                    X_train_split,
                    {'segmentation': masks_train_split, 'classification': y_train_split},
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val_split, {'segmentation': masks_val_split, 'classification': y_val_split}),
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                # Treinamento padrão
                if use_data_augmentation and cnn.data_generator:
                    train_generator = cnn.data_generator.flow(X_train_split, y_train_split, batch_size=batch_size)
                    history = model.fit(
                        train_generator,
                        epochs=epochs,
                        validation_data=(X_val_split, y_val_split),
                        callbacks=callbacks,
                        verbose=1,
                        steps_per_epoch=len(X_train_split) // batch_size
                    )
                else:
                    history = model.fit(
                        X_train_split, y_train_split,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val_split, y_val_split),
                        callbacks=callbacks,
                        verbose=1
                    )
            
            progress_bar.progress(100)
            status_text.text("✅ Treinamento concluído!")
            
            # Salvar no session state
            st.session_state.model = model
            st.session_state.history = history
            st.session_state.class_names = class_names
            st.session_state.cnn = cnn
            st.session_state.use_masks = train_with_masks
            st.session_state.valid_train_indices = valid_train_indices
            
            st.success("🎉 Modelo treinado com sucesso!")
            
        except Exception as e:
            st.error(f"❌ Erro durante o treinamento: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    # Resultados do Treinamento
    if 'model' in st.session_state:
        st.header("📊 Resultados do Treinamento")
        
        # Gráficos de treinamento
        use_masks_trained = st.session_state.get('use_masks', False)
        history_fig = plot_training_history(st.session_state.history, use_masks_trained)
        if history_fig:
            st.plotly_chart(history_fig, use_container_width=True)
        
        # Avaliação no conjunto de teste
        if test_files and len(test_labels) == len(test_files):
            st.subheader("🎯 Avaliação no Conjunto de Teste")
            
            try:
                # Carregar dados de teste
                test_with_masks = use_masks_trained and test_masks and len(test_masks) == len(test_files)
                
                if test_with_masks:
                    X_test, masks_test, y_test, valid_test_indices = load_images_masks_and_labels(
                        test_files, test_masks, test_labels
                    )
                    st.write(f"✅ Teste carregado: {len(X_test)} pares imagem-mask")
                else:
                    X_test, y_test, valid_test_indices = load_images_and_labels(test_files, test_labels)
                    masks_test = None
                    st.write(f"✅ Teste carregado: {len(X_test)} imagens")
                
                # Predições
                if use_masks_trained:
                    predictions = st.session_state.model.predict(X_test, verbose=0)
                    y_pred = np.argmax(predictions[1], axis=1)  # Classificação
                    pred_probs = predictions[1]
                else:
                    predictions = st.session_state.model.predict(X_test, verbose=0)
                    y_pred = np.argmax(predictions, axis=1)
                    pred_probs = predictions
                
                # Calcular métricas de forma segura
                accuracy, precision_avg, recall_avg, cm, report = safe_metrics_calculation(y_test, y_pred, class_names)
                
                if accuracy is not None:
                    # Mostrar métricas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Acurácia", f"{accuracy:.2%}")
                    with col2:
                        st.metric("Precisão Média", f"{precision_avg:.2%}")
                    with col3:
                        st.metric("Recall Médio", f"{recall_avg:.2%}")
                    
                    # Matriz de confusão
                    cm_fig = plot_confusion_matrix_safe(cm, class_names)
                    if cm_fig:
                        st.plotly_chart(cm_fig, use_container_width=True)
                    
                    # Relatório detalhado
                    if report:
                        st.subheader("📋 Relatório Detalhado")
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(3))
                    
                    # Análise detalhada por imagem
                    st.subheader("🔍 Análise Detalhada")
                    
                    # Usar apenas os arquivos válidos (que foram processados com sucesso)
                    valid_test_files = [test_files[i] for i in valid_test_indices]
                    valid_test_labels = [test_labels[i] for i in valid_test_indices]
                    
                    for i, (file, true_label, pred_label) in enumerate(zip(valid_test_files, y_test, y_pred)):
                        with st.expander(f"Imagem {i+1}: {file.name}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                file.seek(0)
                                image = Image.open(file)
                                st.image(image, caption="Imagem Original", use_column_width=True)
                            
                            with col2:
                                st.write(f"**Real:** {class_names[true_label]}")
                                st.write(f"**Predito:** {class_names[pred_label]}")
                                
                                # Probabilidades
                                prob_df = pd.DataFrame({
                                    'Classe': class_names,
                                    'Probabilidade': pred_probs[i]
                                })
                                fig = px.bar(prob_df, x='Classe', y='Probabilidade', 
                                           color='Probabilidade', color_continuous_scale='RdYlGn')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Interpretação
                                max_prob = np.max(pred_probs[i])
                                if pred_label == 2 and max_prob > 0.8:
                                    st.success("🛢️ **ALTA PROBABILIDADE DE RESERVATÓRIO**")
                                elif pred_label == 1 and max_prob > 0.6:
                                    st.warning("⚠️ **POSSÍVEL RESERVATÓRIO**")
                                else:
                                    st.info("ℹ️ **BAIXO POTENCIAL**")
                else:
                    st.error("❌ Não foi possível calcular as métricas")
                
            except Exception as e:
                st.error(f"❌ Erro na avaliação: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    # Predição individual
    st.header("🔮 Análise de Nova Imagem")
    uploaded_image = st.file_uploader("Carregar imagem sísmica", type=['png', 'jpg', 'jpeg'], key="single")
    
    if uploaded_image and 'model' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_image.seek(0)
            image = Image.open(uploaded_image)
            st.image(image, caption="Imagem para análise", use_column_width=True)
        
        with col2:
            try:
                cnn = st.session_state.cnn
                prediction = cnn.predict_single_image(uploaded_image)
                pred_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                st.write(f"**Classificação:** {class_names[pred_class]}")
                st.write(f"**Confiança:** {confidence:.2%}")
                
                # Gráfico
                prob_df = pd.DataFrame({
                    'Classe': class_names,
                    'Probabilidade': prediction[0]
                })
                fig = px.bar(prob_df, x='Classe', y='Probabilidade', 
                           color='Probabilidade', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretação
                if pred_class == 2 and confidence > 0.8:
                    st.success("🛢️ **RESERVATÓRIO PROVÁVEL!**")
                elif pred_class == 1 and confidence > 0.6:
                    st.warning("⚠️ **POSSÍVEL RESERVATÓRIO**")
                else:
                    st.info("ℹ️ **BAIXO POTENCIAL**")
                    
            except Exception as e:
                st.error(f"❌ Erro na predição: {str(e)}")
    
    # Sidebar informações
    with st.sidebar:
        st.header("📖 Informações")
        
        if use_masks:
            st.markdown("""
            **🎯 Modo: Segmentação + Classificação**
            - Arquitetura U-Net
            - Dupla saída (pixel + global)
            - Requer masks 1:1 com imagens
            - Análise pixel-wise
            """)
        elif use_transfer_learning:
            st.markdown("""
            **🚀 Modo: Transfer Learning**
            - Base ResNet50 pré-treinada
            - Fine-tuning para dados sísmicos
            - Convergência mais rápida
            - Melhor generalização
            """)
        else:
            st.markdown("""
            **🧠 Modo: CNN Padrão**
            - 4 blocos convolucionais
            - Global Average Pooling
            - Regularização avançada
            - Dropout inteligente
            """)
        
        st.markdown("""
        **🎨 Classes de Análise:**
        - 🔴 **Não Reservatório**: Estrutura desfavorável
        - 🟡 **Possível Reservatório**: Investigação necessária
        - 🟢 **Provável Reservatório**: Alta probabilidade
        
        **📊 Métricas Calculadas:**
        - Acurácia global
        - Precisão por classe
        - Recall por classe
        - Matriz de confusão
        - F1-Score
        """)
        
        # Status do modelo atual
        if 'model' in st.session_state:
            st.subheader("🤖 Modelo Atual")
            st.write(f"**Parâmetros:** {st.session_state.model.count_params():,}")
            st.write(f"**Camadas:** {len(st.session_state.model.layers)}")
            st.write(f"**Modo:** {'Segmentação' if st.session_state.get('use_masks', False) else 'Classificação'}")

if __name__ == "__main__":
    main()