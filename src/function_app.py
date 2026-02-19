import azure.functions as func
from loguru import logger
from models.model_runner import ModelRunner, MODEL_MAPPINGS

app = func.FunctionApp()

@app.function_name(name="scheduled_model_training")
@app.timer_trigger(
    schedule="0 0 3 1 * *",  # At 03:00:00 UTC on the 1st day of every month (monthly)
    arg_name="timer",
    run_on_startup=False,
    use_monitor=True
)
def scheduled_model_training(timer: func.TimerRequest) -> None:
    """Monthly scheduled training trigger.

    Schedule format: second minute hour day month day-of-week (all UTC).
    Current expression runs at 03:00 UTC on the 1st day of every month.
    Examples:
      Every 15 minutes: 0 */15 * * * *
      At 01:30 UTC daily: 0 30 1 * * *
    use_monitor=True ensures missed executions (during downtime) are replayed once host starts.
    """
    try:
        logger.info("Model Training fired. Starting monthly model training batch.")
        for model in MODEL_MAPPINGS.keys():
            logger.info(f"Starting training for model: {model}")
            ModelRunner(model).run()
        logger.info(f"Model Training completed for model: {model}")
    except Exception as e:
        logger.exception(f"scheduled_model_training failed: {e}")

@app.route(route="model_training_trigger", auth_level=func.AuthLevel.ANONYMOUS)
def model_training_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logger.info('HTTP trigger function processed a request.')
    try:
        req_body = req.get_json()
        logger.info(f"Received from request body: {req_body}")
        name = req_body.get('model_name')
        logger.info(f"Extracted model_name: {name}")
    except Exception as e:
        logger.warning(f"Could not parse JSON body: {e}")
        name = None

    if name:
        logger.info(f"Starting model runner for model: {name}")
        ModelRunner(name).run()
        return func.HttpResponse(f"This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
            f"Could not find {name} in the request.",
            status_code=400
        )
