import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass,  field
from typing import Optional, List, Dict, Any, Tuple
from loguru import logger
from models.utils.helper import Helper
import joblib

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from blob.blob_service import BlobService
from config.settings import Settings
from datetime import datetime


warnings.filterwarnings("ignore")

@dataclass
class ArtfamTrainerConfig():
    tfidf_max_features: int = 100
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_min_df: int = 5
    tfidf_max_df: float = 0.8
    rf_n_estimators: int = 200
    rf_max_depth: int = 20
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 10
    xgb_learning_rate: float = 0.1
    random_state: int = 42
    model_save_path: Path = field(default_factory=lambda: Path(f"artfam_model.pkl"))

class ArtfamRohplatteClassifer(Helper):
    """
    Production-ready trainer for ARTFAM_ROHPLATTE prediction.
    Handles feature engineering, model training, evaluation, and saving.
    """
    def __init__(self, config: Optional[ArtfamTrainerConfig] = None):
        self.config = config or ArtfamTrainerConfig()
        self.tfidf: Optional[TfidfVectorizer] = None
        save_dir: Optional[Path] = None,
        self.klasse_encoder: Optional[LabelEncoder] = None
        self.feature_columns: List[str] = [
            "klasse_encoded",
            "has_dimensions",
            "has_p2",
            "has_p3",
            "has_p5",
            "has_e1",
            "has_mdf",
            "has_u_code",
            "has_w_code",
            "has_r_code",
            "has_optima",
            "has_duropal",
            "has_pfleiderer",
            "has_decoboard",
            "desc_length",
            "desc_word_count",
        ]
        self.model: Optional[Any] = None
        self.model_name: str = ""
        self.non_applicable_classes: List[str] = ["HPL", "CL"]
        self.blob_service = BlobService(container_name="models")
        self.save_dir = save_dir or Path(__file__).parent


    def fit(self, train_df: pd.DataFrame) -> None:
        """Run the full training pipeline."""
        logger.info("Starting model training pipeline for ARTFAM_ROHPLATTE.")
        logger.info(f"Input dataframe shape: {train_df.shape}")
        logger.info(f"Columns: {list(train_df.columns)}")
        X_train, X_test, y_train, y_test = self._prepare_data(training_df=train_df)
        logger.info(f"Prepared features. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        logger.info(f"y_train unique values: {np.unique(y_train)}")
        logger.info(f"y_test unique values: {np.unique(y_test)}")
        models = self._train(X_train, y_train)
        logger.info("Model training complete. Evaluating models...")
        perf = self._evaluate(models, X_train, y_train, X_test, y_test)
        logger.info(f"Best model selected: {perf['best_model_name']}")
        self.model = perf["best_model"]
        self.model_name = perf["best_model_name"]
        self._save()
        logger.info("Model artifacts saved.")

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from description text."""
        df = df.copy()
        df["has_dimensions"] = df["description"].str.contains(r"\d+[Xx]\d+[Xx]?\d*", na=False).astype(int)
        df["has_p2"] = df["description"].str.contains(r"\bP2\b", case=False, na=False).astype(int)
        df["has_p3"] = df["description"].str.contains(r"\bP3\b", case=False, na=False).astype(int)
        df["has_p5"] = df["description"].str.contains(r"\bP5\b", case=False, na=False).astype(int)
        df["has_e1"] = df["description"].str.contains(r"\bE1\b", case=False, na=False).astype(int)
        df["has_mdf"] = df["description"].str.contains(r"\bMDF\b", case=False, na=False).astype(int)
        df["has_u_code"] = df["description"].str.contains(r"\bU\d+", case=False, na=False).astype(int)
        df["has_w_code"] = df["description"].str.contains(r"\bW\d+", case=False, na=False).astype(int)
        df["has_r_code"] = df["description"].str.contains(r"\bR\d+", case=False, na=False).astype(int)
        df["has_optima"] = df["description"].str.contains(r"OPTIMA", case=False, na=False).astype(int)
        df["has_duropal"] = df["description"].str.contains(r"Duropal", case=False, na=False).astype(int)
        df["has_pfleiderer"] = df["description"].str.contains(r"Pfleiderer", case=False, na=False).astype(int)
        df["has_decoboard"] = df["description"].str.contains(r"DecoBoard", case=False, na=False).astype(int)
        df["desc_length"] = df["description"].fillna("").str.len()
        df["desc_word_count"] = df["description"].fillna("").str.split().str.len()
        return df

    def _prepare_data(self, training_df: pd.DataFrame) -> Tuple:
        """Prepare features and targets for training and testing."""
        train_features = self._extract_features(training_df).sample(frac=0.7, random_state=self.config.random_state).reset_index(drop=True)
        test_features = self._extract_features(training_df).sample(frac=0.3, random_state=self.config.random_state).reset_index(drop=True)

        self.tfidf = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=self.config.tfidf_ngram_range,
            min_df=self.config.tfidf_min_df,
            max_df=self.config.tfidf_max_df,
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
        )
        X_train_tfidf = self.tfidf.fit_transform(train_features["description"].fillna(""))
        X_test_tfidf = self.tfidf.transform(test_features["description"].fillna(""))

        self.klasse_encoder = LabelEncoder()
        train_features["klasse_encoded"] = self.klasse_encoder.fit_transform(train_features["klasse"])
        test_features["klasse_encoded"] = self.klasse_encoder.transform(test_features["klasse"])

        X_train_manual = train_features[self.feature_columns].values
        X_test_manual = test_features[self.feature_columns].values

        X_train = hstack([X_train_tfidf, csr_matrix(X_train_manual)])
        X_test = hstack([X_test_tfidf, csr_matrix(X_test_manual)])

        y_train = train_features["artfam_rohplatte"].values
        y_test = test_features["artfam_rohplatte"].values

        # Remove NaNs from targets and features
        if pd.isnull(y_train).any():
            mask = pd.notnull(y_train)
            y_train = y_train[mask]
            X_train = X_train[mask]
        if pd.isnull(y_test).any():
            mask = pd.notnull(y_test)
            y_test = y_test[mask]
            X_test = X_test[mask]

        return X_train, X_test, y_train, y_test

    def _train(self, X_train, y_train) -> Dict[str, Any]:
        """Train Random Forest and XGBoost (if available), return models."""
        rf_model = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=0,
        )
        rf_model.fit(X_train, y_train)

        xgb_model = None

        return {"rf": rf_model, "xgb": xgb_model}

    def _evaluate(self, models: Dict[str, Any], X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Evaluate models and select the best one."""
        rf_model = models["rf"]

        rf_train_pred = rf_model.predict(X_train)
        rf_test_pred = rf_model.predict(X_test)
        rf_train_acc = accuracy_score(y_train, rf_train_pred)
        rf_test_acc = accuracy_score(y_test, rf_test_pred)

        best_model = rf_model
        best_model_name = "Random Forest"
        best_predictions = rf_test_pred

        return {
            "best_model": best_model,
            "best_model_name": best_model_name,
            "best_predictions": best_predictions,
            "rf_train_acc": rf_train_acc,
            "rf_test_acc": rf_test_acc,
        }

    def _save(self) -> None:
        """Save the model and preprocessing objects as a dictionary, then pickle."""
        model_artifacts = {
            "model": self.model,
            "model_name": self.model_name,
            "tfidf_vectorizer": self.tfidf,
            "klasse_encoder": self.klasse_encoder,
            "feature_columns": self.feature_columns,
            "non_applicable_classes": self.non_applicable_classes,
        }
        joblib.dump(model_artifacts, self.config.model_save_path)
        self._save_model_to_blob(model_path = self.config.model_save_path)

