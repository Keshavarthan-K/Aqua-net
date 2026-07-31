import requests
import os

# 1. Paste your OpenTopography API Key here
API_KEY = "619bede3e587154153406128235b5a2e"

# 2. The exact GPS bounding box for the Chennai Metropolitan Area
SOUTH = 12.75
NORTH = 13.30
WEST = 80.10
EAST = 80.35

print("🛰️ Initiating uplink to OpenTopography Global SRTM Database...")

# 3. Construct the API request URL for the NASA SRTM 30m dataset
url = (
    f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1"
    f"&south={SOUTH}&north={NORTH}&west={WEST}&east={EAST}"
    f"&outputFormat=GTiff&API_Key={API_KEY}"
)

print(f"📥 Downloading satellite elevation data for Chennai...")
print("This may take 30-60 seconds. Please wait...")

# 4. Request the file and save it locally
response = requests.get(url)

if response.status_code == 200:
    with open("chennai_elevation.tif", "wb") as file:
        file.write(response.content)
    print("✅ Success! 'chennai_elevation.tif' has been downloaded and saved.")
    print(f"File size: {os.path.getsize('chennai_elevation.tif') / (1024*1024):.2f} MB")
else:
    print(f"❌ Error {response.status_code}: Could not download the file.")
    print(response.text)