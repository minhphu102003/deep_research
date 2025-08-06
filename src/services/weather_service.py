import requests
from schemas.weather_schema import WeatherResult, WeatherForecastResult, WeatherForecast
from open_deep_research.api_keys import (
    ApiKeyEnvVar,
    get_api_key
)


def get_weather(location: str, unit: str = "c") -> WeatherResult:
    response = requests.get(
        "https://api.weatherapi.com/v1/current.json",
        params={"key": get_api_key(ApiKeyEnvVar.WEATHER), "q": location},
        timeout=10
    )
    response.raise_for_status()
    data = response.json()

    temp_key = "temp_c" if unit == "c" else "temp_f"
    feels_like_key = "feelslike_c" if unit == "c" else "feelslike_f"

    return WeatherResult(
        location=location,
        condition=data["current"]["condition"]["text"],
        temperature_c=data["current"][temp_key],
        feels_like_c=data["current"][feels_like_key],
        humidity=data["current"]["humidity"],
        wind_kph=data["current"]["wind_kph"]
    )


def get_weather_forecast(location: str, days: int = 3, unit: str = "c") -> WeatherForecastResult:
    response = requests.get(
        "https://api.weatherapi.com/v1/forecast.json",
        params={
            "key": get_api_key(ApiKeyEnvVar.WEATHER),
            "q": location,
            "days": days
        },
        timeout=10
    )
    response.raise_for_status()
    data = response.json()

    temp_key = "avgtemp_c" if unit == "c" else "avgtemp_f"

    forecast_list = [
        WeatherForecast(
            date=day["date"],
            condition=day["day"]["condition"]["text"],
            avg_temp=day["day"][temp_key]
        )
        for day in data["forecast"]["forecastday"]
    ]

    return WeatherForecastResult(forecast=forecast_list)
