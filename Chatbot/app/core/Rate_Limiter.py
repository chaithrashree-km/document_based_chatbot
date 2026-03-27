from slowapi import Limiter
from slowapi.util import get_remote_address

from app.Config import Config

config = Config()

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=config.REDIS_URL
)