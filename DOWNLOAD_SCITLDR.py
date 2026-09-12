import os
import urllib.request

base_url = "https://raw.githubusercontent.com/allenai/scitldr/master/SciTLDR-Data/SciTLDR-AIC"

output_folder = os.path.join("data", "scitldr")

os.makedirs(output_folder, exist_ok=True)

files = ["train.jsonl", "dev.jsonl", "test.jsonl"]

for filename in files:

    url = f"{base_url}/{filename}"
    output_path = os.path.join(output_folder, filename)

    print(f"Downloading {filename}...")

    urllib.request.urlretrieve(url, output_path)

    print(f"Saved: {output_path}")

print("\nSciTLDR-AIC downloaded successfully!")