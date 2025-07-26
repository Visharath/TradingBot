"""
Model Builder for the Unified CNN-LSTM Trading Bot.

This module implements the unified CNN-LSTM hybrid architecture that combines:
- CNN branches for pattern recognition across multiple kernel sizes
- LSTM branches for sequence modeling with attention mechanism
- Static feature processing for time-based and market regime features
- Advanced fusion layer for intelligent feature combination
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from src.config import Config, ModelConfig
from src.utils import save_model_metadata, save_pickle


class AttentionLayer(layers.Layer):
    """Multi-head attention layer for LSTM outputs."""
    
    def __init__(self, d_model: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads
        )
        self.norm = layers.LayerNormalization()
    
    def call(self, inputs):
        attention_output = self.attention(inputs, inputs)
        return self.norm(inputs + attention_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
        })
        return config


class ModelBuilder:
    """
    Advanced model builder for unified CNN-LSTM trading bot.
    
    The architecture consists of:
    1. CNN Branch: Multi-scale convolutions for pattern recognition
    2. LSTM Branch: Bidirectional LSTM with attention for sequence modeling
    3. Static Branch: Dense layers for non-sequential features
    4. Fusion Layer: Intelligent combination of all branches
    """
    
    def __init__(self, config: Config):
        """
        Initialize the ModelBuilder.
        
        Args:
            config: Configuration object containing model settings
        """
        self.config = config
        self.model_config = config.model
        self.model = None
        self.history = None
        self.feature_names = []
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Configure TensorFlow
        self._configure_tensorflow()
        
        logger.info("ModelBuilder initialized")
    
    def _configure_tensorflow(self):
        """Configure TensorFlow settings for optimal performance."""
        # Enable mixed precision for faster training on compatible hardware
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision enabled")
        except Exception as e:
            logger.warning(f"Could not enable mixed precision: {e}")
        
        # Configure GPU if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
            except RuntimeError as e:
                logger.warning(f"GPU configuration error: {e}")
        else:
            logger.info("No GPUs found, using CPU")
    
    def build_model(
        self,
        input_shape: Tuple[int, int],
        n_classes: int = 3,
        static_features: int = 20
    ) -> keras.Model:
        """
        Build the unified CNN-LSTM model architecture.
        
        Args:
            input_shape: Shape of sequential input (sequence_length, n_features)
            n_classes: Number of output classes (default: 3 for Buy/Hold/Sell)
            static_features: Number of static (non-sequential) features
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building model with input_shape={input_shape}, n_classes={n_classes}")
        
        # Input layers
        sequence_input = layers.Input(shape=input_shape, name='sequence_input')
        static_input = layers.Input(shape=(static_features,), name='static_input')
        
        # CNN Branch - Multi-scale pattern recognition
        cnn_outputs = []
        for i, (filters, kernel_size) in enumerate(zip(
            self.model_config.cnn_filters,
            self.model_config.cnn_kernel_sizes
        )):
            # Convolutional layers with different kernel sizes
            conv = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i}_filters_{filters}_kernel_{kernel_size}'
            )(sequence_input)
            
            # Batch normalization
            conv = layers.BatchNormalization(name=f'bn_conv_{i}')(conv)
            
            # Max pooling
            conv = layers.MaxPooling1D(
                pool_size=2,
                name=f'maxpool_{i}'
            )(conv)
            
            # Dropout
            conv = layers.Dropout(
                self.model_config.dropout_rate,
                name=f'dropout_conv_{i}'
            )(conv)
            
            cnn_outputs.append(conv)
        
        # Concatenate CNN outputs
        if len(cnn_outputs) > 1:
            cnn_combined = layers.Concatenate(name='cnn_concat')(cnn_outputs)
        else:
            cnn_combined = cnn_outputs[0]
        
        # Global average pooling for CNN
        cnn_features = layers.GlobalAveragePooling1D(name='cnn_global_pool')(cnn_combined)
        
        # LSTM Branch - Sequence modeling
        lstm_input = sequence_input
        
        # Stack bidirectional LSTM layers
        for i, units in enumerate(self.model_config.lstm_units):
            return_sequences = i < len(self.model_config.lstm_units) - 1
            
            lstm_input = layers.Bidirectional(
                layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.model_config.dropout_rate,
                    recurrent_dropout=self.model_config.dropout_rate,
                    name=f'lstm_{i}_units_{units}'
                ),
                name=f'bidirectional_lstm_{i}'
            )(lstm_input)
            
            # Layer normalization
            if return_sequences:
                lstm_input = layers.LayerNormalization(name=f'ln_lstm_{i}')(lstm_input)
        
        # Attention mechanism for LSTM
        if len(self.model_config.lstm_units) > 1:
            # Reshape for attention if we have sequence output
            lstm_reshaped = layers.Reshape((-1, self.model_config.lstm_units[-1] * 2))(lstm_input)
            attention_output = AttentionLayer(
                d_model=self.model_config.lstm_units[-1] * 2,
                num_heads=self.model_config.attention_heads,
                name='attention_layer'
            )(lstm_reshaped)
            lstm_features = layers.GlobalAveragePooling1D(name='lstm_global_pool')(attention_output)
        else:
            lstm_features = lstm_input
        
        # Static Features Branch
        static_branch = layers.Dense(
            64,
            activation='relu',
            name='static_dense_1'
        )(static_input)
        static_branch = layers.BatchNormalization(name='static_bn_1')(static_branch)
        static_branch = layers.Dropout(
            self.model_config.dropout_rate,
            name='static_dropout_1'
        )(static_branch)
        
        static_branch = layers.Dense(
            32,
            activation='relu',
            name='static_dense_2'
        )(static_branch)
        static_branch = layers.BatchNormalization(name='static_bn_2')(static_branch)
        static_features_out = layers.Dropout(
            self.model_config.dropout_rate,
            name='static_dropout_2'
        )(static_branch)
        
        # Fusion Layer - Combine all branches
        fusion_input = layers.Concatenate(name='fusion_concat')([
            cnn_features,
            lstm_features,
            static_features_out
        ])
        
        # Dense fusion layers
        fusion = layers.Dense(
            256,
            activation='relu',
            name='fusion_dense_1'
        )(fusion_input)
        fusion = layers.BatchNormalization(name='fusion_bn_1')(fusion)
        fusion = layers.Dropout(
            self.model_config.dropout_rate,
            name='fusion_dropout_1'
        )(fusion)
        
        fusion = layers.Dense(
            128,
            activation='relu',
            name='fusion_dense_2'
        )(fusion)
        fusion = layers.BatchNormalization(name='fusion_bn_2')(fusion)
        fusion = layers.Dropout(
            self.model_config.dropout_rate,
            name='fusion_dropout_2'
        )(fusion)
        
        fusion = layers.Dense(
            64,
            activation='relu',
            name='fusion_dense_3'
        )(fusion)
        fusion = layers.BatchNormalization(name='fusion_bn_3')(fusion)
        fusion = layers.Dropout(
            self.model_config.dropout_rate,
            name='fusion_dropout_3'
        )(fusion)
        
        # Output layer
        if n_classes == 2:
            # Binary classification
            output = layers.Dense(
                1,
                activation='sigmoid',
                name='output_binary'
            )(fusion)
        else:
            # Multi-class classification
            output = layers.Dense(
                n_classes,
                activation='softmax',
                name='output_multiclass'
            )(fusion)
        
        # Create model
        model = keras.Model(
            inputs=[sequence_input, static_input],
            outputs=output,
            name='unified_cnn_lstm_trading_bot'
        )
        
        # Compile model
        self._compile_model(model, n_classes)
        
        # Print model summary
        model.summary(print_fn=logger.info)
        
        self.model = model
        return model
    
    def _compile_model(self, model: keras.Model, n_classes: int):
        """
        Compile the model with appropriate loss function and metrics.
        
        Args:
            model: Keras model to compile
            n_classes: Number of output classes
        """
        # Choose loss function based on number of classes
        if n_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy', 'sparse_categorical_accuracy']
        
        # Choose optimizer
        optimizer = optimizers.Adam(
            learning_rate=self.model_config.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with {loss} loss and {metrics} metrics")
    
    def prepare_data(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        sequence_length: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Prepare data for training by creating sequences and separating static features.
        
        Args:
            features: DataFrame with all features
            targets: Series with target labels
            sequence_length: Length of sequences (uses config default if None)
            
        Returns:
            Tuple of (input_dict, targets_array)
        """
        if sequence_length is None:
            sequence_length = self.model_config.sequence_length
        
        logger.info(f"Preparing data with sequence_length={sequence_length}")
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        # Identify static features (time-based features that don't change much)
        static_feature_patterns = ['hour', 'minute', 'day_of_week', 'month', 'quarter', 'is_monday', 'is_friday']
        static_features = [col for col in features.columns if any(pattern in col for pattern in static_feature_patterns)]
        sequence_features = [col for col in features.columns if col not in static_features]
        
        logger.info(f"Static features: {len(static_features)}, Sequence features: {len(sequence_features)}")
        
        # Convert to numpy arrays
        features_array = features.values
        targets_array = targets.values
        
        # Create sequences
        X_sequences = []
        X_static = []
        y = []
        
        for i in range(sequence_length, len(features_array)):
            # Sequential features
            seq_data = features_array[i-sequence_length:i, [features.columns.get_loc(col) for col in sequence_features]]
            X_sequences.append(seq_data)
            
            # Static features (use current time point)
            static_data = features_array[i, [features.columns.get_loc(col) for col in static_features]]
            X_static.append(static_data)
            
            # Target
            y.append(targets_array[i])
        
        X_sequences = np.array(X_sequences)
        X_static = np.array(X_static)
        y = np.array(y)
        
        logger.info(f"Created {len(X_sequences)} sequences")
        logger.info(f"Sequence shape: {X_sequences.shape}, Static shape: {X_static.shape}, Targets shape: {y.shape}")
        
        # Prepare input dictionary
        inputs = {
            'sequence_input': X_sequences,
            'static_input': X_static
        }
        
        return inputs, y
    
    def train(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        validation_split: Optional[float] = None,
        save_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Train the model with the provided data.
        
        Args:
            features: DataFrame with features
            targets: Series with target labels
            validation_split: Fraction of data to use for validation
            save_path: Path to save the trained model
            
        Returns:
            Dictionary with training metrics
        """
        if validation_split is None:
            validation_split = self.model_config.validation_split
        
        logger.info("Starting model training...")
        
        # Prepare data
        inputs, y = self.prepare_data(features, targets)
        
        # Determine input shapes
        sequence_shape = inputs['sequence_input'].shape[1:]  # (sequence_length, n_features)
        static_shape = inputs['static_input'].shape[1]  # (n_static_features,)
        n_classes = len(np.unique(y))
        
        # Build model if not already built
        if self.model is None:
            self.build_model(sequence_shape, n_classes, static_shape)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            [inputs['sequence_input'], inputs['static_input']],
            y,
            test_size=validation_split,
            random_state=42,
            stratify=y
        )
        
        train_inputs = {'sequence_input': X_train[0], 'static_input': X_train[1]}
        val_inputs = {'sequence_input': X_val[0], 'static_input': X_val[1]}
        
        # Callbacks
        callbacks_list = self._create_callbacks(save_path)
        
        # Train model
        self.history = self.model.fit(
            train_inputs,
            y_train,
            batch_size=self.model_config.batch_size,
            epochs=self.model_config.epochs,
            validation_data=(val_inputs, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluate model
        train_loss, train_acc = self.model.evaluate(train_inputs, y_train, verbose=0)
        val_loss, val_acc = self.model.evaluate(val_inputs, y_val, verbose=0)
        
        # Generate predictions for detailed metrics
        train_pred = self.model.predict(train_inputs)
        val_pred = self.model.predict(val_inputs)
        
        if n_classes > 2:
            train_pred_classes = np.argmax(train_pred, axis=1)
            val_pred_classes = np.argmax(val_pred, axis=1)
        else:
            train_pred_classes = (train_pred > 0.5).astype(int).flatten()
            val_pred_classes = (val_pred > 0.5).astype(int).flatten()
        
        # Calculate metrics
        metrics = {
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'train_detailed_accuracy': float(accuracy_score(y_train, train_pred_classes)),
            'val_detailed_accuracy': float(accuracy_score(y_val, val_pred_classes)),
        }
        
        logger.info("Training completed!")
        logger.info(f"Training accuracy: {train_acc:.4f}")
        logger.info(f"Validation accuracy: {val_acc:.4f}")
        
        # Print classification report
        logger.info("Validation Classification Report:")
        logger.info(f"\\n{classification_report(y_val, val_pred_classes, target_names=self.model_config.class_names[:n_classes])}")
        
        # Save model and metadata
        if save_path:
            self.save_model(save_path, metrics)
        
        return metrics
    
    def _create_callbacks(self, save_path: Optional[Path]) -> List[keras.callbacks.Callback]:
        """Create training callbacks."""
        callbacks_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.model_config.patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.model_config.patience // 2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        
        # Model checkpoint
        if save_path:
            checkpoint = callbacks.ModelCheckpoint(
                save_path / 'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks_list.append(checkpoint)
        
        return callbacks_list
    
    def predict(
        self,
        features: pd.DataFrame,
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using the trained model.
        
        Args:
            features: DataFrame with features
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predictions array or tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Prepare data
        inputs, _ = self.prepare_data(features, pd.Series([0] * len(features)))
        
        # Make predictions
        probabilities = self.model.predict(inputs)
        
        if len(probabilities.shape) == 2 and probabilities.shape[1] > 1:
            # Multi-class
            predictions = np.argmax(probabilities, axis=1)
        else:
            # Binary
            predictions = (probabilities > 0.5).astype(int).flatten()
            if len(probabilities.shape) == 2:
                probabilities = probabilities.flatten()
        
        if return_probabilities:
            return predictions, probabilities
        else:
            return predictions
    
    def save_model(self, save_path: Path, metrics: Dict[str, float]):
        """
        Save the trained model and metadata.
        
        Args:
            save_path: Directory to save the model
            metrics: Training metrics
        """
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = save_path / 'model.h5'
        self.model.save(model_file)
        
        # Save feature names
        feature_file = save_path / 'feature_names.pkl'
        save_pickle(self.feature_names, feature_file)
        
        # Save training history
        history_file = save_path / 'training_history.pkl'
        save_pickle(self.history.history if self.history else {}, history_file)
        
        # Save model metadata
        save_model_metadata(
            model_file,
            self.config,
            metrics,
            self.feature_names,
            0  # Training time not tracked here
        )
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: Path) -> bool:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load model
            model_file = model_path / 'model.h5'
            self.model = keras.models.load_model(
                model_file,
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            
            # Load feature names
            feature_file = model_path / 'feature_names.pkl'
            if feature_file.exists():
                from src.utils import load_pickle
                self.feature_names = load_pickle(feature_file)
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """
        Plot training history.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / 'training_history.png', dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Path] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.model_config.class_names[:len(np.unique(y_true))],
            yticklabels=self.model_config.class_names[:len(np.unique(y_true))]
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()