"""
Weather Application using Open-Meteo (FREE, No API Key)

- Uses Open-Meteo Geocoding API
- Uses Open-Meteo Weather API
- Safe error handling
- Always returns dictionary
"""

import requests


def get_weather_data(city: str) -> dict:
    """
    Fetch current weather for a given city using Open-Meteo.
    Always returns a dictionary and never raises exceptions.
    """

    try:
        # ------------------ Step 1: Geocoding ------------------
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        geo_response = requests.get(geo_url, params=geo_params, timeout=10)
        geo_data = geo_response.json()

        if "results" not in geo_data or not geo_data["results"]:
            return {
                "city": city,
                "error": "City not found",
                "source": "open-meteo"
            }

        location = geo_data["results"][0]
        latitude = location["latitude"]
        longitude = location["longitude"]

        # ------------------ Step 2: Weather ------------------
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": True
        }

        weather_response = requests.get(weather_url, params=weather_params, timeout=10)
        weather_data = weather_response.json()

        current = weather_data.get("current_weather", {})

        return {
            "city": city,
            "latitude": latitude,
            "longitude": longitude,
            "temperature_c": current.get("temperature"),
            "windspeed_kmh": current.get("windspeed"),
            "winddirection": current.get("winddirection"),
            "weathercode": current.get("weathercode"),
            "time": current.get("time"),
            "source": "open-meteo"
        }

    except Exception as e:
        return {
            "city": city,
            "error": "Failed to fetch weather data",
            "details": str(e),
            "source": "open-meteo"
        }


# ------------------ MAIN ------------------

if __name__ == "__main__":
    city_name = input("Enter city name: ").strip()

    if not city_name:
        print("City name cannot be empty.")
    else:
        result = get_weather_data(city_name)
        print("\nWeather Result:")
        print(result)
