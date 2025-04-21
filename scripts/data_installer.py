import argparse
import requests

def download_file(url, output_path):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
    print(f"Download complete: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Download a file from a URL.")
    parser.add_argument("--url", required=True, help="The URL of the file to download.")
    parser.add_argument("--output", required=True, help="The path to save the downloaded file.")
    
    args = parser.parse_args()
    download_file(args.url, args.output)

if __name__ == "__main__":
    main()


# Data installation
# HE data
# python -m scripts.data_installer --url https://zenodo.org/records/10066853/files/HE_zenodo.zip?download=1 --output /root/BiOSGen/data/HE.zip

# NKX3 data
# python -m scripts.data_installer --url https://zenodo.org/records/10066853/files/NKX3_zenodo.zip?download=1 --output /root/BiOSGen/data/NKX3.zip

# Tissue mask
# python -m scripts.data_installer --url https://figshare.com/ndownloader/files/51023048 --output /root/BiOSGen/data/tissue_mask.zip

# Bbox
# python -m scripts.data_installer --url https://figshare.com/ndownloader/files/51023045 --output /root/BiOSGen/data/bbox.zip