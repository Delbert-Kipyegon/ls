import requests
from requests.auth import HTTPBasicAuth
import pandas as pd

 
url = ""
username = "webservice"
password = "iqZwQDaYj665WV0aOgbSYFCDHsT9GxSxOUTTwOr5IV0="


response = requests.get(url, auth=HTTPBasicAuth(username, password))

 
if response.status_code == 200:
    data = response.json()['value']
    df = pd.DataFrame(data)
    df.to_csv('Quote_details.csv', index = False)
    print(df.head())
else:
    print(f"Failed to fetch data: {response.status_code}")
