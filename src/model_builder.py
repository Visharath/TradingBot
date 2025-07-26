"""
Unified CNN-LSTM Model Builder for the Trading Bot.

This module implements the hybrid CNN-LSTM architecture with:
- CNN Branch: Multi-scale convolutions
- LSTM Branch: Bidirectional LSTM with Multi-Head Attention  
- Static Branch: Time-based and market regime features
- Fusion Layer: Intelligent feature combination
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, callbacks
from tensorflow.keras.utils import plot_model
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path

from .config import config
from .utils import logger, timing_decorator, save_model_artifacts


class UnifiedModelBuilder:
    """
    Builder class for the unified CNN-LSTM trading model.
    
    The model architecture consists of three main branches:
    1. CNN Branch: Processes sequential patterns with multiple kernel sizes
    2. LSTM Branch: Captures long-term dependencies with attention
    3. Static Branch: Processes time-based and market regime features
    
    These branches are then fused and fed through dense layers for final prediction.
    """
    
    def __init__(self):
        self.model = None
        self.history = None
        self.callbacks_list = []
        
    @timing_decorator
    def build_model(
        self,
        input_shape: Tuple[int, int],
        static_features_dim: int = 10
    ) -> Model:
        """
        Build the unified CNN-LSTM model architecture.
        
        Args:
            input_shape: Shape of sequential input (sequence_length, n_features)
            static_features_dim: Number of static features
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building unified CNN-LSTM model with input shape: {input_shape}")
        
        # Input layers
        sequence_input = layers.Input(shape=input_shape, name='sequence_input')
        static_input = layers.Input(shape=(static_features_dim,), name='static_input')
        
        # === CNN BRANCH ===
        cnn_branch = self._build_cnn_branch(sequence_input)
        
        # === LSTM BRANCH ===
        lstm_branch = self._build_lstm_branch(sequence_input)
        
        # === STATIC BRANCH ===
        static_branch = self._build_static_branch(static_input)
        
        # === FUSION LAYER ===
        fused_features = self._build_fusion_layer(cnn_branch, lstm_branch, static_branch)
        
        # === OUTPUT LAYER ===
        output = self._build_output_layer(fused_features)
        
        # Create model
        self.model = Model(
            inputs=[sequence_input, static_input],
            outputs=output,
            name='unified_cnn_lstm_trading_bot'
        )
        
        # Compile model
        self._compile_model()
        
        logger.info(f"Model built successfully with {self.model.count_params():,} parameters")
        
        return self.model
    
    def _build_cnn_branch(self, sequence_input: layers.Input) -> layers.Layer:
        """
        Build CNN branch with multi-scale convolutions.
        
        Args:
            sequence_input: Sequential input tensor
            
        Returns:
            CNN branch output layer
        """
        # Multi-scale CNN paths
        cnn_outputs = []
        
        for i, (filters, kernel_size) in enumerate(
            zip(config.model.cnn_filters, config.model.cnn_kernel_sizes)
        ):
            # Convolutional path
            conv = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation=config.model.cnn_activation,
                padding='same',
                name=f'cnn_conv1d_{i}_kernel_{kernel_size}'
            )(sequence_input)
            
            # Batch normalization
            conv = layers.BatchNormalization(name=f'cnn_batch_norm_{i}')(conv)
            
            # Dropout
            conv = layers.Dropout(
                config.model.cnn_dropout,
                name=f'cnn_dropout_{i}'
            )(conv)
            
            # Additional conv layer for depth
            conv = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation=config.model.cnn_activation,
                padding='same',
                name=f'cnn_conv1d_{i}_2_kernel_{kernel_size}'
            )(conv)
            
            # Global max pooling
            pooled = layers.GlobalMaxPooling1D(
                name=f'cnn_global_max_pool_{i}'
            )(conv)
            
            cnn_outputs.append(pooled)
        
        # Concatenate multi-scale features
        if len(cnn_outputs) > 1:
            cnn_combined = layers.Concatenate(name='cnn_multi_scale_concat')(cnn_outputs)
        else:
            cnn_combined = cnn_outputs[0]
        
        # Dense layer to compress CNN features
        cnn_dense = layers.Dense(
            128,
            activation='relu',
            name='cnn_dense_compression'
        )(cnn_combined)
        
        cnn_output = layers.Dropout(
            config.model.cnn_dropout,
            name='cnn_final_dropout'
        )(cnn_dense)
        
        return cnn_output
    
    def _build_lstm_branch(self, sequence_input: layers.Input) -> layers.Layer:
        """
        Build LSTM branch with bidirectional LSTM and multi-head attention.
        
        Args:
            sequence_input: Sequential input tensor
            
        Returns:
            LSTM branch output layer
        """
        # Input normalization
        normalized_input = layers.LayerNormalization(name='lstm_input_norm')(sequence_input)
        
        # Bidirectional LSTM layer
        if config.model.bidirectional:
            lstm_layer = layers.Bidirectional(
                layers.LSTM(
                    config.model.lstm_units,
                    return_sequences=True,
                    dropout=config.model.lstm_dropout,
                    recurrent_dropout=config.model.lstm_recurrent_dropout,
                    name='lstm_main'
                ),
                name='bidirectional_lstm'
            )(normalized_input)
        else:
            lstm_layer = layers.LSTM(
                config.model.lstm_units,
                return_sequences=True,
                dropout=config.model.lstm_dropout,
                recurrent_dropout=config.model.lstm_recurrent_dropout,
                name='lstm_main'
            )(normalized_input)
        
        # Layer normalization after LSTM
        lstm_normalized = layers.LayerNormalization(name='lstm_output_norm')(lstm_layer)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=config.model.attention_heads,
            key_dim=config.model.attention_key_dim,
            name='multi_head_attention'
        )(lstm_normalized, lstm_normalized)
        
        # Add & Norm (Residual connection)
        attention_residual = layers.Add(name='attention_residual')([lstm_normalized, attention_output])
        attention_normalized = layers.LayerNormalization(name='attention_norm')(attention_residual)
        
        # Global average pooling to get fixed-size output
        lstm_pooled = layers.GlobalAveragePooling1D(name='lstm_global_avg_pool')(attention_normalized)
        
        # Dense layer
        lstm_dense = layers.Dense(
            128,
            activation='relu',
            name='lstm_dense'
        )(lstm_pooled)
        
        lstm_output = layers.Dropout(
            config.model.lstm_dropout,
            name='lstm_final_dropout'
        )(lstm_dense)
        
        return lstm_output
    
    def _build_static_branch(self, static_input: layers.Input) -> layers.Layer:
        """
        Build static branch for time-based and market regime features.
        
        Args:
            static_input: Static features input tensor
            
        Returns:
            Static branch output layer
        """
        static_layer = static_input
        
        # Dense layers for static features
        for i, units in enumerate(config.model.static_dense_units):
            static_layer = layers.Dense(
                units,
                activation='relu',
                name=f'static_dense_{i}'
            )(static_layer)
            
            static_layer = layers.BatchNormalization(
                name=f'static_batch_norm_{i}'
            )(static_layer)
            
            static_layer = layers.Dropout(
                config.model.static_dropout,
                name=f'static_dropout_{i}'
            )(static_layer)
        
        return static_layer
    
    def _build_fusion_layer(
        self,
        cnn_features: layers.Layer,
        lstm_features: layers.Layer,
        static_features: layers.Layer
    ) -> layers.Layer:
        """
        Build fusion layer to combine all branch outputs.
        
        Args:
            cnn_features: CNN branch output
            lstm_features: LSTM branch output  
            static_features: Static branch output
            
        Returns:
            Fused features layer
        """
        # Concatenate all features
        fused = layers.Concatenate(name='feature_fusion')([
            cnn_features,
            lstm_features,
            static_features
        ])
        
        # Dense layers for feature fusion
        for i, units in enumerate(config.model.fusion_dense_units):
            fused = layers.Dense(
                units,
                activation='relu',
                name=f'fusion_dense_{i}'
            )(fused)
            
            fused = layers.BatchNormalization(
                name=f'fusion_batch_norm_{i}'
            )(fused)
            
            fused = layers.Dropout(
                config.model.fusion_dropout,
                name=f'fusion_dropout_{i}'
            )(fused)
        
        return fused
    
    def _build_output_layer(self, fused_features: layers.Layer) -> layers.Layer:
        """
        Build output layer for classification.
        
        Args:
            fused_features: Fused features from all branches
            
        Returns:
            Output layer with predictions
        """
        output = layers.Dense(
            config.model.num_classes,
            activation=config.model.output_activation,
            name='classification_output'
        )(fused_features)
        
        return output
    
    def _compile_model(self) -> None:
        """Compile the model with optimizer, loss, and metrics."""
        # Choose optimizer
        if config.model.optimizer.lower() == "adam":
            optimizer = optimizers.Adam(learning_rate=config.model.learning_rate)
        elif config.model.optimizer.lower() == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=config.model.learning_rate)
        elif config.model.optimizer.lower() == "sgd":
            optimizer = optimizers.SGD(learning_rate=config.model.learning_rate)
        else:
            optimizer = optimizers.Adam(learning_rate=config.model.learning_rate)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=config.model.loss_function,
            metrics=config.model.metrics
        )
    
    def _setup_callbacks(self, model_save_path: str) -> List[callbacks.Callback]:
        """
        Set up training callbacks.
        
        Args:
            model_save_path: Path to save the best model
            
        Returns:
            List of Keras callbacks
        """
        callback_list = []
        
        # Model checkpoint
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callback_list.append(checkpoint_callback)
        
        # Early stopping
        early_stop_callback = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.model.patience,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop_callback)
        
        # Reduce learning rate on plateau
        reduce_lr_callback = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.model.patience // 2,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr_callback)
        
        # TensorBoard logging
        log_dir = config.LOGS_DIR / "tensorboard"
        log_dir.mkdir(exist_ok=True)
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callback_list.append(tensorboard_callback)
        
        # Custom logging callback
        logging_callback = self._create_logging_callback()
        callback_list.append(logging_callback)
        
        return callback_list
    
    def _create_logging_callback(self) -> callbacks.Callback:
        """Create custom logging callback."""
        class LoggingCallback(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                logger.info(
                    f"Epoch {epoch + 1}: "
                    f"loss={logs.get('loss', 0):.4f}, "
                    f"accuracy={logs.get('accuracy', 0):.4f}, "
                    f"val_loss={logs.get('val_loss', 0):.4f}, "
                    f"val_accuracy={logs.get('val_accuracy', 0):.4f}"
                )
        
        return LoggingCallback()
    
    @timing_decorator
    def train_model(
        self,
        X_train: Tuple[np.ndarray, np.ndarray],
        y_train: np.ndarray,
        X_val: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        y_val: Optional[np.ndarray] = None,
        model_save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the unified model.
        
        Args:
            X_train: Tuple of (sequence_data, static_data) for training
            y_train: Training targets
            X_val: Tuple of (sequence_data, static_data) for validation
            y_val: Validation targets
            model_save_path: Path to save the trained model
            
        Returns:
            Training history and metrics
        """
        if self.model is None:
            raise ValueError("Model must be built before training")
        
        logger.info("Starting model training...")
        
        # Set default save path
        if model_save_path is None:
            model_save_path = str(config.MODELS_DIR / "unified_trading_model.h5")
        
        # Setup callbacks
        self.callbacks_list = self._setup_callbacks(model_save_path)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train model
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=config.model.batch_size,
            epochs=config.model.epochs,
            validation_data=validation_data,
            validation_split=config.model.validation_split if validation_data is None else None,
            callbacks=self.callbacks_list,
            verbose=1
        )
        
        # Save training artifacts
        self._save_training_artifacts(model_save_path)
        
        logger.info("Model training completed successfully")
        
        return {
            "history": self.history.history,
            "model_path": model_save_path,
            "best_val_loss": min(self.history.history.get('val_loss', [float('inf')])),
            "best_val_accuracy": max(self.history.history.get('val_accuracy', [0]))
        }
    
    def _save_training_artifacts(self, model_path: str) -> None:
        """Save model and training metadata."""
        # Prepare metadata
        metadata = {
            "model_architecture": "unified_cnn_lstm",
            "model_config": config.model.__dict__,
            "training_history": self.history.history if self.history else {},
            "model_summary": self._get_model_summary(),
            "total_parameters": self.model.count_params(),
            "trainable_parameters": sum([
                tf.keras.backend.count_params(w) for w in self.model.trainable_weights
            ])
        }
        
        # Save model and metadata
        metadata_path = model_path.replace('.h5', '_metadata.json')
        save_model_artifacts(self.model, metadata, model_path, metadata_path)
        
        # Save model architecture plot
        try:
            plot_path = model_path.replace('.h5', '_architecture.png')
            plot_model(
                self.model,
                to_file=plot_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                dpi=150
            )
            logger.info(f"Model architecture plot saved to {plot_path}")
        except Exception as e:
            logger.warning(f"Could not save model architecture plot: {e}")
    
    def _get_model_summary(self) -> str:
        """Get string representation of model summary."""
        if self.model is None:
            return ""
        
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)
    
    def predict(
        self,
        X: Tuple[np.ndarray, np.ndarray],
        return_probabilities: bool = True
    ) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Tuple of (sequence_data, static_data)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before making predictions")
        
        predictions = self.model.predict(X, verbose=0)
        
        if not return_probabilities:
            return np.argmax(predictions, axis=1)
        
        return predictions
    
    def evaluate(
        self,
        X_test: Tuple[np.ndarray, np.ndarray],
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before evaluation")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics_dict = {}
        for i, metric_name in enumerate(['loss'] + config.model.metrics):
            metrics_dict[metric_name] = results[i]
        
        return metrics_dict
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            print("No model built yet")
            return
        
        self.model.summary()


def create_sequences(
    data: np.ndarray,
    sequence_length: int,
    prediction_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    
    Args:
        data: Input data array
        sequence_length: Length of input sequences
        prediction_horizon: Steps ahead to predict
        
    Returns:
        Tuple of (X, y) sequences
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length + prediction_horizon - 1])
    
    return np.array(X), np.array(y)


def prepare_model_inputs(
    sequence_data: np.ndarray,
    static_data: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
    test_split: float = 0.2,
    val_split: float = 0.2
) -> Dict[str, Tuple[np.ndarray, ...]]:
    """
    Prepare inputs for model training.
    
    Args:
        sequence_data: Sequential features
        static_data: Static features
        targets: Target values
        sequence_length: Length of sequences
        test_split: Fraction for test set
        val_split: Fraction for validation set (from training set)
        
    Returns:
        Dictionary containing train/val/test splits
    """
    # Create sequences
    X_seq, y_seq = create_sequences(sequence_data, sequence_length)
    
    # Align static data with sequences
    X_static = static_data[sequence_length:]
    y_aligned = targets[sequence_length:]
    
    # Train/test split
    split_idx = int(len(X_seq) * (1 - test_split))
    
    X_seq_train, X_seq_test = X_seq[:split_idx], X_seq[split_idx:]
    X_static_train, X_static_test = X_static[:split_idx], X_static[split_idx:]
    y_train, y_test = y_aligned[:split_idx], y_aligned[split_idx:]
    
    # Train/validation split
    val_split_idx = int(len(X_seq_train) * (1 - val_split))
    
    X_seq_train_final = X_seq_train[:val_split_idx]
    X_seq_val = X_seq_train[val_split_idx:]
    X_static_train_final = X_static_train[:val_split_idx]
    X_static_val = X_static_train[val_split_idx:]
    y_train_final = y_train[:val_split_idx]
    y_val = y_train[val_split_idx:]
    
    return {
        "train": ((X_seq_train_final, X_static_train_final), y_train_final),
        "val": ((X_seq_val, X_static_val), y_val),
        "test": ((X_seq_test, X_static_test), y_test)
    }


__all__ = [
    "UnifiedModelBuilder",
    "create_sequences", 
    "prepare_model_inputs"
]