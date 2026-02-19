# Pfleiderer Model Training - Azure Functions

A serverless Azure Functions application for automated machine learning model training, designed to train and manage multiple classification models on a monthly schedule.

## Overview

This project implements an automated model training pipeline using Azure Functions with Python. It supports multiple classification models including:

- **ARTFAM_ROH**: Article Family and Raw Plate Classifier
- **KLASSE_ROH**: Raw Material Classifier  
- **KLASSE_CORE**: Core Material Classifier

The application runs on a scheduled basis (monthly) and can also be triggered on-demand via HTTP requests.

## Features

- 🔄 **Scheduled Training**: Automatic monthly model retraining (1st of every month at 03:00 UTC)
- 🌐 **HTTP Triggers**: On-demand training via HTTP API
- 🗄️ **PostgreSQL Integration**: Data retrieval from PostgreSQL database
- 📦 **Azure Blob Storage**: Model persistence and versioning
- 🔔 **Slack Notifications**: Real-time training status updates
- 🐳 **Docker Support**: Containerized deployment for consistency
- 🔧 **Local Development**: Docker Compose setup for local testing

## Tech Stack

- **Runtime**: Azure Functions (Python 3.11)
- **ML Libraries**: scikit-learn, XGBoost, PyTorch, transformers, sentence-transformers
- **Database**: PostgreSQL (via SQLAlchemy, psycopg2)
- **Storage**: Azure Blob Storage
- **Logging**: Loguru
- **CI/CD**: Bitbucket Pipelines
- **Container Registry**: Azure Container Registry (ACR)

## Project Structure

```
pfleiderer-fa-model-training/
├── src/
│   ├── function_app.py          # Azure Functions entry point
│   ├── host.json                # Functions host configuration
│   ├── blob/                    # Blob storage services
│   │   └── blob_service.py
│   ├── config/                  # Configuration management
│   │   └── settings.py
│   ├── database/                # Database services
│   │   └── service.py
│   ├── models/                  # Model training logic
│   │   ├── model_runner.py      # Model execution orchestrator
│   │   ├── models_interface.py  # Base model interface
│   │   ├── training_queries.py  # SQL queries for training data
│   │   ├── artfam_rohplatte/    # Article family classifier
│   │   └── core/                # Core classifiers
│   │       ├── klasse/          # Material class models
│   │       └── roh/             # Raw material models
│   └── utils/                   # Utility functions
│       └── notify_slack.py
├── volumes/                     # Local development volumes
│   ├── azurite_data/           # Local blob storage emulator
│   └── postgres/               # Local PostgreSQL data
├── dockerfile                   # Container image definition
├── docker-compose.yml          # Local development setup
├── bitbucket-pipelines.yml     # CI/CD pipeline
└── requirements.txt            # Python dependencies
```

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Azure Functions Core Tools (for local development)
- Azure CLI (for deployment)
- PostgreSQL database
- Azure Storage Account

## Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Environment
ENVIRONMENT=LOCAL  # or DEV, STAGING, PROD

# Azure Storage
AzureWebJobsStorage=<connection-string>

# PostgreSQL
POSTGRES_CONN_STR=postgresql://user:password@host:port/database

# Azure Identity
UAMI_CLIENT_ID=<user-assigned-managed-identity-client-id>

# Email Storage Account (for model artifacts)
EMAIL_STORAGE_ACCOUNT_NAME=<storage-account-name>
EMAIL_STORAGE_ACCOUNT_KEY=<storage-account-key>
EMAIL_STORAGE_ACCOUNT_URL=https://<account-name>.blob.core.windows.net

# Slack Notifications (optional)
ENABLE_SLACK=True
SLACK_TOKEN=<slack-bot-token>
```

## Local Development

### Using Docker Compose

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pfleiderer-fa-model-training
   ```

2. **Create `.env` file** with required environment variables

3. **Start the services**
   ```bash
   docker-compose up -d
   ```

   This starts:
   - PostgreSQL database (port 5432)
   - Azurite storage emulator (ports 10000-10002)
   - Azure Functions app (port 7071)

4. **Access the Function App**
   - Local endpoint: `http://localhost:7071`
   - Azure Storage Explorer can connect to Azurite at `http://localhost:10000`

5. **Trigger training manually**
   ```bash
   curl -X POST http://localhost:7071/api/model_training_trigger \
     -H "Content-Type: application/json" \
     -d '{"model_name": "ARTFAM_ROH"}'
   ```

### Without Docker

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Azure Functions locally**
   ```bash
   cd src
   func start
   ```

## API Endpoints

### HTTP Trigger - Manual Model Training

**Endpoint**: `/api/model_training_trigger`  
**Method**: POST  
**Auth Level**: Anonymous (configure appropriately for production)

**Request Body**:
```json
{
  "model_name": "ARTFAM_ROH"
}
```

**Available Models**:
- `ARTFAM_ROH`
- `KLASSE_ROH`
- `KLASSE_CORE`

**Response**:
```
This HTTP triggered function executed successfully.
```

### Timer Trigger - Scheduled Training

**Schedule**: `0 0 3 1 * *` (03:00 UTC on the 1st of every month)  
**Behavior**: Trains all models sequentially  
**Monitor**: Enabled (replays missed executions)

## Deployment

### CI/CD Pipeline (Bitbucket Pipelines)

The project uses Bitbucket Pipelines for automated deployment:

**Branches**:
- `develop` → Deploy to Development ACR (`pflacrdevuks001`)
- `staging` → Deploy to Staging ACR (`pflacrstguks001`)
- `main` → Deploy to Production ACR (`pflacrproduks001`)

**Pipeline Steps**:
1. Preflight checks
2. Build Docker image (linux/amd64)
3. Push to Azure Container Registry
4. Auto-deploy to Azure Function App (configured in Azure)

**Required Secrets** (configured in Bitbucket):
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`
- `AZURE_TENANT_ID`
- `AZURE_SUBSCRIPTION_ID`

### Manual Deployment

1. **Build Docker image**
   ```bash
   docker build -t pfl-fa-model-train:latest --platform linux/amd64 .
   ```

2. **Login to Azure Container Registry**
   ```bash
   az acr login --name <acr-name>
   ```

3. **Tag and push image**
   ```bash
   docker tag pfl-fa-model-train:latest <acr-name>.azurecr.io/pfl-fa-model-train:latest
   docker push <acr-name>.azurecr.io/pfl-fa-model-train:latest
   ```

4. **Update Function App** to use the new image

## Model Training Flow

1. **Data Retrieval**: Fetch training data from PostgreSQL using predefined queries
2. **Model Training**: Execute model-specific training logic
3. **Model Validation**: Validate trained models
4. **Model Storage**: Save trained models to Azure Blob Storage
5. **Notification**: Send status updates via Slack (if enabled)
6. **Logging**: Comprehensive logging via Loguru

## Monitoring & Logging

- **Application Insights**: Monitor function executions and performance
- **Loguru**: Structured logging throughout the application
- **Slack Notifications**: Real-time updates on training status
- **Health Checks**: PostgreSQL health monitoring in Docker Compose

## Development Guidelines

### Adding a New Model

1. Create model class inheriting from the base interface in `src/models/`
2. Add training query in `src/models/training_queries.py`
3. Register model in `MODEL_MAPPINGS` in `src/models/model_runner.py`
4. Implement required methods: `fit()`, `predict()`, `save()`, `load()`

### Testing

```bash
# Run unit tests (configure test framework as needed)
pytest tests/

# Test specific model
curl -X POST http://localhost:7071/api/model_training_trigger \
  -H "Content-Type: application/json" \
  -d '{"model_name": "YOUR_MODEL_NAME"}'
```

## Troubleshooting

### Common Issues

**Function not starting**:
- Check environment variables are properly set
- Verify PostgreSQL connection string
- Ensure Azure Storage connection is valid

**Model training fails**:
- Check database connection and query validity
- Verify sufficient data is available
- Review logs in Application Insights or container logs

**Docker build fails**:
- Ensure platform is set to `linux/amd64`
- Check for dependency conflicts in `requirements.txt`

**View logs**:
```bash
# Docker Compose
docker-compose logs -f azure-function-app

# Specific container
docker logs model-training-function-app
```

## Performance Considerations

- Models are trained sequentially in scheduled runs
- Training data is fetched once per model
- Consider memory constraints when training large models
- Azure Functions timeout: 10 minutes (Premium plan recommended for longer training)

## Security

- Use **Managed Identity** for Azure resource access
- Store secrets in **Azure Key Vault** (recommended for production)
- Configure **appropriate authentication levels** for HTTP triggers
- Use **private endpoints** for database connections in production
- Regularly update dependencies for security patches

## Contributing

1. Create a feature branch from `develop`
2. Make your changes
3. Test locally using Docker Compose
4. Submit a pull request to `develop`
5. After review, changes will be promoted through staging to main

## License

[Specify your license]

## Support

For issues or questions, please contact [team contact information]

---

**Last Updated**: November 2025  
**Maintained by**: QuantSpark Team
