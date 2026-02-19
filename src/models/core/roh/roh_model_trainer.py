"""
ROH classification model training pipeline using n-gram vectorization and MLP.

This module provides a production-ready class for training binary classification models
to identify ROH (raw material) products from text descriptions.
"""

import pickle
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics import BinaryAccuracy, BinaryF1Score
from blob.blob_service import BlobService
from config.settings import Settings
from models.utils.helper import Helper
from models.models_interface import ModelTrainerInterface
from datetime import datetime




@dataclass
class VectorizationConfig:
    """Configuration for text vectorization parameters."""
    
    ngram_range: Tuple[int, int] = (1, 2)
    top_k: int = 20000
    token_mode: str = "word"
    min_document_frequency: int = 2


@dataclass
class ModelConfig:
    """Configuration for MLP model architecture and training."""
    
    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 128
    layers: int = 2
    units: int = 64
    dropout_rate: float = 0.0
    patience: int = 2
    device: str = "cpu"
    model_output_path: Path = field(default_factory=lambda: Path(f"roh_model.pth"))
    holdout_df_path: Path = field(default_factory=lambda: Path(f"roh_holdout_df.pkl"))
    selector_output_path: Path = field(default_factory=lambda: Path(f"roh_selector.pkl"))
    vectorizer_output_path: Path = field(default_factory=lambda: Path(f"roh_vectorizer.pkl"))

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    loss_tracker: List[float]
    accuracy_tracker: List[float]
    f1_tracker: List[float]
    epoch_tracker: List[int]
    best_val_loss: float
    final_accuracy: float
    final_loss: float


class MLPModel(nn.Module):
    """Multi-layer perceptron model for binary classification."""
    
    def __init__(
        self,
        input_shape: int,
        layers: int,
        units: int,
        dropout_rate: float,
    ) -> None:
        """
        Initialize the MLP model.
        
        Args:
            input_shape: Dimension of input features.
            layers: Number of hidden layers.
            units: Number of units in each hidden layer.
            dropout_rate: Dropout probability for regularization.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_shape, units))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        for _ in range(layers - 1):
            self.layers.append(nn.Linear(units, units))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        self.layers.append(nn.BatchNorm1d(units))
        self.layers.append(nn.Linear(units, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor of shape (batch_size,).
        """
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)


class RohClassifier(Helper, ModelTrainerInterface):
    """
    Training pipeline for ROH (raw material) binary classification.
    
    This class handles the complete ML pipeline including data preprocessing,
    feature engineering, model training, and evaluation for identifying ROH products.
    
    Attributes:
        save_dir: Directory for saving models and artifacts.
        roh_classes: List of class labels considered as ROH.
        vectorization_config: Configuration for text vectorization.
        model_config: Configuration for model architecture and training.
        vectorizer: Fitted TF-IDF vectorizer.
        selector: Fitted feature selector.
        model: Trained PyTorch model.
    """
    
    def __init__(
        self,
        save_dir: Optional[Path] = None,
        roh_classes: Optional[List[str]] = None,
        vectorization_config: Optional[VectorizationConfig] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        """
        Initialize the ROH model trainer.
        
        Args:
            save_dir: Directory to save models and artifacts. Defaults to script directory.
            roh_classes: List of class labels to consider as ROH.
            vectorization_config: Text vectorization parameters.
            model_config: Model architecture and training parameters.
        """
        self.save_dir = save_dir or Path(__file__).parent
        self.roh_classes = roh_classes or ["ROH_STAN", "ROH_SOND", "ROH_FIXM"]
        self.vectorization_config = vectorization_config or VectorizationConfig()
        self.model_config = model_config or ModelConfig()
        self.blob_service = BlobService(container_name="models")
        
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.selector: Optional[SelectKBest] = None
        self.model: Optional[MLPModel] = None
        self.metrics: Optional[TrainingMetrics] = None

    @staticmethod
    def clean_descriptions(text: str) -> str:
        """
        Clean and normalize text descriptions.
        
        Args:
            text: Raw text description.
            
        Returns:
            Cleaned text with punctuation removed, lowercased, and no newlines.
        """
        if not isinstance(text, str):
            return ""
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.lower()
        return text.replace("\n", "")
    
    def fit(self, df: pd.DataFrame, num_rows_to_holdout: int = 50) -> Tuple[float, float]:
        """
        Complete training pipeline from raw data to trained model.
        
        Args:
            df: Input dataframe with Description and klasse columns.
            num_rows_to_holdout: Number of rows to set aside for holdout testing.
            
        Returns:
            Tuple of (final_accuracy, final_loss).
        """
        # Data preprocessing
        df = self._clean_df(df)
        df = self._set_aside_holdout_df(df, num_rows_to_holdout)
        
        # Train-test split
        X_train, X_test, y_train, y_test = self._create_test_train_split(df)
        
        # Train model
        val_acc, val_loss = self._train_ngram_model(
            data=((X_train, y_train), (X_test, y_test))
        )
        
        return val_acc, val_loss

    
    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataframe and add ROH binary labels.
        
        Args:
            df: Input dataframe with Description and klasse columns.
            
        Returns:
            Cleaned dataframe with 'roh' binary column added.
        """
        df["description"] = df["description"].apply(self.clean_descriptions)
        df["roh"] = df["klasse"].isin(self.roh_classes)
        return df
    
    def _set_aside_holdout_df(
        self,
        df: pd.DataFrame,
        num_rows_to_holdout: int = 50,
    ) -> pd.DataFrame:
        """
        Set aside a holdout dataset and save it to disk.
        
        Args:
            df: Input dataframe.
            num_rows_to_holdout: Number of rows to hold out for final testing.
            
        Returns:
            Remaining dataframe after holdout removed.
        """
        holdout_df = df[:num_rows_to_holdout]
        holdout_path = self.model_config.holdout_df_path
        joblib.dump(holdout_df, holdout_path)
        self._save_model_to_blob(model_path=holdout_path)
        return df[num_rows_to_holdout:]
    
    def _create_test_train_split(
        self, df: pd.DataFrame,
    ) -> Tuple[List[str], List[str], List[bool], List[bool]]:
        """
        Split data into train and test sets.
        
        Args:
            df: Input dataframe with Description and roh columns.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        X = df["description"].to_list()
        y = df["roh"].to_list()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        return X_train, X_test, y_train, y_test
    
    def ngram_vectorize(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorize texts as n-gram vectors using TF-IDF.
        
        Each text is converted to a TF-IDF vector based on vocabulary of unigrams and bigrams.
        
        Args:
            train_texts: Training text strings.
            train_labels: Training labels for feature selection.
            val_texts: Validation text strings.
            
        Returns:
            Tuple of (x_train, x_val) as vectorized numpy arrays.
        """
        kwargs = {
            "ngram_range": self.vectorization_config.ngram_range,
            "dtype": "int32",
            "strip_accents": "unicode",
            "decode_error": "replace",
            "analyzer": self.vectorization_config.token_mode,
            "min_df": self.vectorization_config.min_document_frequency,
        }
        self.vectorizer = TfidfVectorizer(**kwargs)
        
        # Learn vocabulary from training texts and vectorize
        x_train = self.vectorizer.fit_transform(train_texts)
        
        # Vectorize validation texts
        x_val = self.vectorizer.transform(val_texts)
        
        # Select top K features
        self.selector = SelectKBest(
            f_classif, 
            k=min(self.vectorization_config.top_k, x_train.shape[1])
        )
        self.selector.fit(x_train, train_labels)
        x_train = self.selector.transform(x_train).astype("float32")
        x_val = self.selector.transform(x_val).astype("float32")
        
        # Save artifacts
        joblib.dump(self.vectorizer, self.model_config.vectorizer_output_path)
        self._save_model_to_blob(model_path=self.model_config.vectorizer_output_path)
        joblib.dump(self.selector, self.model_config.selector_output_path)
        self._save_model_to_blob(model_path=self.model_config.selector_output_path)
        
        return x_train, x_val
    
    def _validate_labels(
        self, 
        train_labels: List[bool], 
        val_labels: List[bool]
    ) -> None:
        """
        Validate that validation labels are consistent with training labels.
        
        Args:
            train_labels: Training labels.
            val_labels: Validation labels.
            
        Raises:
            ValueError: If validation labels contain unexpected values.
        """
        num_classes = len(set(train_labels))
        unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
        if len(unexpected_labels):
            msg = (
                "Unexpected label values found in the validation set:"
                f" {unexpected_labels}. Please make sure that the "
                "labels in the validation set are in the same range "
                "as training labels."
            )
            raise ValueError(msg)
    
    def _prepare_data_loaders(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        train_labels: List[bool],
        val_labels: List[bool],
    ) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
        """
        Prepare PyTorch data loaders from numpy arrays.
        
        Args:
            x_train: Training features.
            x_val: Validation features.
            train_labels: Training labels.
            val_labels: Validation labels.
            
        Returns:
            Tuple of (train_loader, val_loader, train_labels_tensor, val_labels_tensor).
        """
        # Convert sparse matrices to dense
        x_train = x_train.todense()
        x_val = x_val.todense()
        
        # Convert to PyTorch tensors
        x_train_tensor = torch.FloatTensor(x_train)
        x_val_tensor = torch.FloatTensor(x_val)
        train_labels_tensor = torch.FloatTensor(train_labels)
        val_labels_tensor = torch.FloatTensor(val_labels)
        
        # Create datasets and loaders
        train_dataset = TensorDataset(x_train_tensor, train_labels_tensor)
        val_dataset = TensorDataset(x_val_tensor, val_labels_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.model_config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.model_config.batch_size
        )
        
        return train_loader, val_loader, train_labels_tensor, val_labels_tensor
    
    def _calculate_pos_weight(self, train_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Calculate positive class weight for handling class imbalance.
        
        Args:
            train_labels: Training labels tensor.
            device: PyTorch device.
            
        Returns:
            Positive weight tensor for BCEWithLogitsLoss.
        """
        count_roh = np.array(train_labels).sum()
        count_non_roh = len(train_labels) - count_roh
        pos_weight = 1 - (count_non_roh / (count_non_roh + count_roh))
        return torch.tensor(pos_weight).to(torch.float).to(device=device)
    
    def _train_epoch(
        self,
        model: MLPModel,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ) -> None:
        """
        Train model for one epoch.
        
        Args:
            model: PyTorch model.
            train_loader: Training data loader.
            criterion: Loss function.
            optimizer: Optimizer.
            device: PyTorch device.
        """
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    def _evaluate_epoch(
        self,
        model: MLPModel,
        val_loader: DataLoader,
        criterion: nn.Module,
        accuracy: BinaryAccuracy,
        f1: BinaryF1Score,
        device: torch.device,
    ) -> Tuple[float, float, float]:
        """
        Evaluate model on validation set.
        
        Args:
            model: PyTorch model.
            val_loader: Validation data loader.
            criterion: Loss function.
            accuracy: Accuracy metric.
            f1: F1 score metric.
            device: PyTorch device.
            
        Returns:
            Tuple of (val_loss, accuracy_score, f1_score).
        """
        model.eval()
        val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                predicted = (outputs > 0).float()
                accuracy.update(predicted, batch_y)
                f1.update(predicted, batch_y)
                num_batches += 1
            val_loss /= num_batches
        
        accuracy_score = accuracy.compute()
        f1_score = f1.compute()
        f1.reset()
        accuracy.reset()
        
        return val_loss, accuracy_score.item(), f1_score.item()
    
    def _train_ngram_model(
        self,
        data: Tuple[Tuple[List[str], List[bool]], Tuple[List[str], List[bool]]],
    ) -> Tuple[float, float]:
        """
        Train n-gram model on the given dataset.
        
        Args:
            data: Tuple of ((train_texts, train_labels), (val_texts, val_labels)).
            
        Returns:
            Tuple of (final_accuracy, final_loss).
            
        Raises:
            ValueError: If validation data has unexpected label values.
        """
        # Get the data
        (train_texts, train_labels), (val_texts, val_labels) = data
        
        # Validate labels
        self._validate_labels(train_labels, val_labels)
        
        # Vectorize texts
        x_train, x_val = self.ngram_vectorize(train_texts, train_labels, val_texts)
        
        # Prepare data loaders
        train_loader, val_loader, train_labels_tensor, val_labels_tensor = \
            self._prepare_data_loaders(x_train, x_val, train_labels, val_labels)
        
        # Initialize model
        self.model = MLPModel(
            input_shape=x_train.shape[1],
            layers=self.model_config.layers,
            units=self.model_config.units,
            dropout_rate=self.model_config.dropout_rate,
        )
        
        # Setup device and metrics
        device = torch.device(self.model_config.device)
        accuracy = BinaryAccuracy().to(device)
        f1 = BinaryF1Score().to(device)
        
        # Setup loss and optimizer
        pos_weight = self._calculate_pos_weight(train_labels_tensor, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.model_config.learning_rate)
        
        self.model.to(device)
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        loss_tracker = []
        accuracy_tracker = []
        f1_tracker = []
        epoch_tracker = []
        
        for epoch in range(self.model_config.epochs):
            # Train one epoch
            self._train_epoch(self.model, train_loader, criterion, optimizer, device)
            
            # Evaluate
            val_loss, accuracy_score, f1_score = self._evaluate_epoch(
                self.model, val_loader, criterion, accuracy, f1, device
            )
            
            # Track metrics
            loss_tracker.append(val_loss)
            accuracy_tracker.append(accuracy_score)
            f1_tracker.append(f1_score)
            epoch_tracker.append(epoch)
            
            logger.info(
                f"Epoch {epoch+1}/{self.model_config.epochs}, "
                f"accuracy: {accuracy_score:.4f}, "
                f"loss: {val_loss:.4f}, "
                f"f1: {f1_score:.4f}"
            )
            
            # Save best model and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                model_path = self.model_config.model_output_path
                logger.info(f"Saving best model to {model_path}")
                torch.save(self.model.state_dict(), model_path)
                self._save_model_to_blob(model_path=model_path)
            else:
                patience_counter += 1
                if patience_counter >= self.model_config.patience:
                    logger.info("Early stopping")
                    break
        
        # Store metrics
        self.metrics = TrainingMetrics(
            loss_tracker=loss_tracker,
            accuracy_tracker=accuracy_tracker,
            f1_tracker=f1_tracker,
            epoch_tracker=epoch_tracker,
            best_val_loss=best_val_loss,
            final_accuracy=accuracy_score,
            final_loss=val_loss,
        )
        
        return accuracy_score, val_loss

