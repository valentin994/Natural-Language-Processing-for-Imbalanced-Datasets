import requests


def check_if_country(country_name):
    url = f"https://parseapi.back4app.com/classes/Continentscountriescities_Country?limit=10&keys=name&where=%0A%7B%0A++++%22name%22%3A+%22{country_name}%22%0A%7D%0A"

    headers = {
        "X-Parse-Application-Id": "2ZYCYxF06gVr5QKn0ka5gRQ7Q4LZpBjKcyPJDAp8",
        "X-Parse-REST-API-Key": "HHgWIh32y7LKrogyboljnDKoN7rGlv2GlNyrNDqX",
    }
    r = requests.get(url, headers=headers)
    return r.json()
