from models.artfam_rohplatte.artfam_model_classifier import ArtfamRohplatteClassifer
from models.core.roh.roh_model_trainer import RohClassifier
from models.core.klasse.core_classifier import CoreClassifier
from models.training_queries import ARTFAM_QUERY, CORE_QUERY, ROH_QUERY
from database.service import DatabaseService
from loguru import logger
from utils.notify_slack import send_slack_update
from config.settings import Settings

MODEL_MAPPINGS = {
    "ARTFAM_ROH": {"model_classifier": ArtfamRohplatteClassifer, "query": ARTFAM_QUERY},
    "KLASSE_ROH": {"model_classifier": RohClassifier, "query": ROH_QUERY},
    "KLASSE_CORE": {"model_classifier": CoreClassifier, "query": CORE_QUERY},
}

class ModelRunner:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.db_service = DatabaseService()

    def run(self):
        if self.model_name not in MODEL_MAPPINGS:
            raise ValueError(f"Model {self.model_name} not found in mappings.")
        try:
            model_info = MODEL_MAPPINGS.get(self.model_name)
            model_trainer_class = model_info.get("model_classifier", None)
            message = f"Starting training for model: {self.model_name} in {Settings.ENVIRONMENT}."
            send_slack_update(message)

            logger.info(f"Retrieved model trainer for {self.model_name}: {model_trainer_class}")
            query = model_info.get("query", None)
            if model_trainer_class is None or query is None:
                raise ValueError(f"Model classifier or query not defined for model {self.model_name}.")
            logger.info(f"Fetching data for model {self.model_name} using query.")

            df = self.db_service.fetch_data(query)
            logger.info(f"Data fetched for model {self.model_name}, records: {len(df)}.")
            model_trainer = model_trainer_class()
            model_trainer.fit(df)
            completed_message = f"Model {self.model_name} training completed in {Settings.ENVIRONMENT}."

            send_slack_update(completed_message)
            logger.info(completed_message)
        except Exception as e:
            message = f"Error occurred while running model in {Settings.ENVIRONMENT} for {self.model_name}, Error: {e}"
            send_slack_update(message)
            logger.exception(message)
            raise
        
        