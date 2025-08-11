from enum import Enum
import os

class ApiKeyEnvVar(str, Enum):
    SERPAPI = "SERPAPI_API_KEY"
    WEATHER = "WEATHER_API_KEY"

def get_api_key(env_var: ApiKeyEnvVar) -> str:
    api_key = os.getenv(env_var.value)
    if not api_key:
        raise ValueError(f"{env_var.value} environment variable is not set.")
    return api_key