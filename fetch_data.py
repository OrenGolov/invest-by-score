import json
import urllib.request

import pandas as pd

ticker = "AAPL"
url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
params = "?range=6mo&interval=1d"
req = urllib.request.Request(url + params, headers={"User-Agent": "Mozilla/5.0"})

with urllib.request.urlopen(req, timeout=10) as resp:
    payload = json.load(resp)

result = payload["chart"]["result"][0]
timestamps = result["timestamp"]
quote = result["indicators"]["quote"][0]

data = pd.DataFrame(
    {
        "Open": quote["open"],
        "High": quote["high"],
        "Low": quote["low"],
        "Close": quote["close"],
        "Volume": quote["volume"],
    },
    index=pd.to_datetime(timestamps, unit="s"),
)

print(data.tail())