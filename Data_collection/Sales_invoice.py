import requests
from requests.auth import HTTPBasicAuth
import pandas as pd

url = "https://bctest.dayliff.com:7048/BC160/ODataV4/Company('KENYA')/Sales_Invoice?$top=5000"
username = "webservice"
password = "iqZwQDaYj665WV0aOgbSYFCDHsT9GxSxOUTTwOr5IV0="

headers = {
    "Accept": "application/json"
}

response = requests.get(url, auth=HTTPBasicAuth(username, password), headers=headers)

if response.status_code == 200:
    data = response.json().get('value', [])
    if data:
        df = pd.DataFrame(data)
        df.to_csv('Sales_Invoice.csv', index=False)
        print(df.head())
    else:
        print("API responded but no data returned.")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
