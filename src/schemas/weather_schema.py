from pydantic import BaseModel, Field
from typing import TypedDict, Literal

class CurrentWeatherInput(BaseModel):
    location: str = Field(..., description="Name of the city or location to get the current weather.")
    unit: Literal["c", "f"] = Field("c", description="Temperature unit: 'c' for Celsius, 'f' for Fahrenheit.")

class WeatherResult(TypedDict):
    location: str
    condition: str
    temperature_c: float
    feels_like_c: float
    humidity: int
    wind_kph: float

class ForecastWeatherInput(BaseModel):
    location: str = Field(..., description="Name of the city or location to get the weather forecast.")
    days: int = Field(3, ge=1, le=7, description="Number of days to forecast (1-7).")
    unit: Literal["c", "f"] = Field("c", description="Temperature unit: 'c' for Celsius, 'f' for Fahrenheit.")

class WeatherForecast(BaseModel):
    date: str
    condition: str
    avg_temp: float

class WeatherForecastResult(BaseModel):
    forecast: list[WeatherForecast]
