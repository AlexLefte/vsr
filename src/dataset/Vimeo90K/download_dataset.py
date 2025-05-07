import requests
from tqdm import tqdm  # Import tqdm for progress bar

def download_zip(url, save_path):
    # Send GET request to the URL
    response = requests.get(url, stream=True)  # stream=True for downloading in chunks
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the total file size from the response headers
        total_size = int(response.headers.get('Content-Length', 0))
        
        # Open the local file in write-binary mode
        with open(save_path, 'wb') as file:
            # Create a progress bar using tqdm
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as bar:
                # Iterate over the content in chunks
                for chunk in response.iter_content(chunk_size=1024):  # Download in 1KB chunks
                    if chunk:
                        file.write(chunk)  # Write each chunk to file
                        bar.update(len(chunk))  # Update the progress bar
        print(f"File downloaded successfully: {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

# Example usage
url = "http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip"  # Replace with your actual URL
save_path = "datasets/Vimeo90K/raw/dataset.zip"  # Local path where the file will be saved

download_zip(url, save_path)
