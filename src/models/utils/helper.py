from loguru import logger
from pathlib import Path
from datetime import datetime

class Helper:
    def _save_model_to_blob(self, model_path: Path, container_name: str = "models") -> None:
        """
        Save the trained model file to Azure Blob Storage.
        Archives the existing model to 'archived/' folder before uploading the new one.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        
        # Archive existing model if it exists
        self._archive_existing_model(model_path.name)
        
        # Upload new model
        with open(model_path, "rb") as f:
            model_data = f.read()
        logger.info(f"Uploading model to blob storage: {container_name}/{model_path.name}")
        self.blob_service.upload_blob(
            blob_name=model_path.name,
            data=model_data
        )
    
    def _archive_existing_model(self, blob_name: str) -> None:
        """
        Archive the existing model to 'archived/' folder with timestamp.
        """
        try:
            # Check if the blob exists
            container_client = self.blob_service.container_client
            blob_client = container_client.get_blob_client(blob_name)
            
            if blob_client.exists():
                # Download existing model
                existing_model_data = blob_client.download_blob().readall()
                
                # Create archived blob name with timestamp
                timestamp = datetime.now().strftime("%Y%m%d")
                archived_blob_name = f"archived/{timestamp}_{blob_name}"
                
                logger.info(f"Archiving existing model: {blob_name} -> {archived_blob_name}")
                
                # Upload to archived location
                self.blob_service.upload_blob(
                    blob_name=archived_blob_name,
                    data=existing_model_data
                )
            else:
                logger.info(f"No existing model to archive: {blob_name}")
        except Exception as e:
            logger.warning(f"Failed to archive existing model {blob_name}: {e}")