import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
assert WANDB_API_KEY, "Please put your w and b api key"
logger.debug("Api key loaded")
earning_prob = [0.2, 0.8, 0.4]
eps = 0.1
