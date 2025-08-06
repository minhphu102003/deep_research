from langchain_core.tools import tool
from schemas.weather_schema import (
    CurrentWeatherInput,
    WeatherResult,
    ForecastWeatherInput,
    WeatherForecastResult
)
from services.weather_service import get_weather, get_weather_forecast

@tool(args_schema=CurrentWeatherInput)
def get_current_weather(location: str, unit: str = "c") -> WeatherResult:
    """Get current weather at a location."""
    return get_weather(location, unit)

@tool(args_schema=ForecastWeatherInput)
def get_weather_forecast(location: str, days: int = 3, unit: str = "c") -> WeatherForecastResult:
    """Get the weather forecast for a location for the next 1–7 days."""
    return get_weather_forecast(location, days, unit)