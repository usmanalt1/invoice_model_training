"""
Production-ready Core Classifier for Wood Product Descriptions
Trains and evaluates multiple ML models to predict product core types from descriptions
"""

from typing import Optional, Dict, Tuple, List, Any, Union
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from models.utils.helper import Helper
from models.models_interface import ModelTrainerInterface
from blob.blob_service import BlobService
from dataclasses import dataclass, field

logger = logging.getLogger("core_classifier")
logging.basicConfig(level=logging.INFO)

@dataclass
class CoreClassifierConfig():
    model_output_path: Path = field(default_factory=lambda: Path("core_classifier_model.pkl"))
    vectorizer_output_path: Path = field(default_factory=lambda: Path("tfidf_vectorizer.pkl"))
    

class CoreClassifier(Helper, ModelTrainerInterface):
    """
    A class for training core type classification models using TF-IDF and multiple classifiers.
    Mimics the structure and modularity of KlasseModelTrainer.
    """
    def __init__(self,
                 target_col: str = "core",
                 min_samples: int = 4,
                 vectorizer_params: Optional[Dict[str, Any]] = None):
        self.target_col = target_col
        self.min_samples = min_samples
        self.vectorizer_params = vectorizer_params or {
            "max_features": 5000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.8,
            "strip_accents": "unicode",
            "lowercase": True,
        }
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[Any] = None
        self.model_name: Optional[str] = None
        self.le_name_mapping: Optional[Dict] = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.results: Optional[Dict[str, Any]] = None
        self.blob_service = BlobService(container_name="models")
        self.config = CoreClassifierConfig() 

    def fit(self, df: pd.DataFrame, text_column: str = "description") -> "CoreClassifier":
        """
        Complete training pipeline: preprocess, filter, feature extraction, train, evaluate.
        """
        df_filtered = self._filter_data(df)
        X_train, X_test, y_train, y_test = self._prepare_train_test(df_filtered, text_column)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.results = self._train_and_evaluate_models(X_train, X_test, y_train, y_test)
        self.model = self.results[self.model_name]["model"]
        self._save_model()
        return self

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        value_counts = df[self.target_col].value_counts()
        filtered_values = value_counts[value_counts >= self.min_samples].index
        df_filtered = df[df[self.target_col].isin(filtered_values)]
        return df_filtered

    def _prepare_train_test(self, df: pd.DataFrame, text_column: str) -> Tuple[Any, Any, Any, Any]:
        X = df[text_column].fillna("").apply(str).apply(str.lower)
        y = df[self.target_col]
        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)
        X_tfidf = self.vectorizer.fit_transform(X)
        # Simple split (can be replaced with stratified split if needed)
        n = len(df)
        split = int(n * 0.7)
        X_train, X_test = X_tfidf[:split], X_tfidf[split:]
        y_train, y_test = y[:split], y[split:]
        return X_train, X_test, y_train, y_test

    def _train_and_evaluate_models(self, X_train, X_test, y_train, y_test) -> Dict[str, Dict[str, Any]]:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial"),
            "Naive Bayes": MultinomialNB(),
            "Linear SVM": LinearSVC(max_iter=2000, random_state=42),
        }
        param_grids = {
            "Logistic Regression": {"C": [0.1, 1.0, 10.0], "solver": ["lbfgs", "saga"]},
            "Naive Bayes": {"alpha": [0.1, 0.5, 1.0]},
            "Linear SVM": {"C": [0.1, 1.0, 10.0]},
        }
        results = {}
        for model_name, model in models.items():
            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=5,
                scoring="accuracy",
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
            y_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            results[model_name] = {
                "model": best_model,
                "best_params": grid_search.best_params_,
                "cv_score": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "test_accuracy": test_accuracy,
                "predictions": y_pred,
            }
            logger.info(f"{model_name} | Best params: {grid_search.best_params_} | CV: {cv_scores.mean():.4f} | Test: {test_accuracy:.4f}")
        self.model_name = max(results, key=lambda x: results[x]["test_accuracy"])
        return results

    def _save_model(self) -> None:
        model_path = Path(self.config.model_output_path)
        vectorizer_path = Path(self.config.vectorizer_output_path)
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")
        self._save_model_to_blob(model_path = model_path)
        self._save_model_to_blob(model_path = vectorizer_path)
