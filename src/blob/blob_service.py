from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient        
from azure.identity import ManagedIdentityCredential
from config.settings import Settings

class BlobService:
    def __init__(self, container_name: str):
        self.container_name = container_name
        env = Settings.ENVIRONMENT
        if env == "LOCAL":
            azurite_connection_string = (
            "DefaultEndpointsProtocol=http;"
                f"AccountName={Settings.EMAIL_STORAGE_ACCOUNT_NAME};"
                f"AccountKey={Settings.EMAIL_STORAGE_ACCOUNT_KEY};"
                f"BlobEndpoint={Settings.EMAIL_STORAGE_ACCOUNT_URL}/{Settings.EMAIL_STORAGE_ACCOUNT_NAME};"
            )
            self.blob_service_client = BlobServiceClient.from_connection_string(azurite_connection_string)
        else:
            credential = ManagedIdentityCredential(client_id=Settings.UAMI_CLIENT_ID)

            self.blob_service_client = BlobServiceClient(
                account_url=Settings.EMAIL_STORAGE_ACCOUNT_URL, credential=credential
            )
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.container_client.

    
    def upload_blob(self, blob_name: str, data: bytes):
        self.container_client.upload_blob(name=blob_name, data=data, overwrite=True)