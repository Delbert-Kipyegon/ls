import requests
from requests.auth import HTTPBasicAuth

url = "https://bctest.dayliff.com/BC160/?company=KENYA&bookmark=12%3bfsQAAACHyBcB&page=50870&dc=0&signInRedirected=1&runinframe=1#"
username = "webservice"
password = "iqZwQDaYj665WV0aOgbSYFCDHsT9GxSxOUTTwOr5IV0="

 
response = requests.get(url, auth=HTTPBasicAuth(username, password))

 
if response.status_code == 200:
    print("Response received:")
    try:
        data = response.json()
        print("JSON Response:")
        print(data) 
    except requests.exceptions.JSONDecodeError:
        print("Error: Failed to decode JSON. The response might not be in JSON format.")
        print("Raw Response:")
        print(response.text) 
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
