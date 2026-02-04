#!/usr/bin/env python
# coding: utf-8


import requests
import pandas as pd
from tqdm import tqdm
import time


# Adjust for full dataset
NUM_ENTRIES = 17385  # Start with a small number for testing.

output_json= rf"/depot/home/username/Scraped_Data_{NUM_ENTRIES}.json" # path to output file in json format
output_csv= rf"/depot/home/username/Scraped_Data_{NUM_ENTRIES}.csv"  # path to output file in json format


# Store all LNP data
lnp_data = []

for i in tqdm(range(1, NUM_ENTRIES + 1)):
    url = f"https://lnpdb.molcube.com:8000/search/textsearch/{i}/"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            lnp_data.append(data)
        else:
            print(f"Skipped ID {i} — status {response.status_code}")
    except Exception as e:
        print(f"Error with ID {i}: {e}")

    time.sleep(0.2)  # Prevent overloading the server

# Save as JSON file (preserves structure)
with open(output_json, "w") as f:
    import json
    json.dump(lnp_data, f, indent=2)

# Optional: Save as flat CSV
df = pd.json_normalize(lnp_data)
df.to_csv(output_csv, index=False)

print("Done. Saved JSON and CSV.")





