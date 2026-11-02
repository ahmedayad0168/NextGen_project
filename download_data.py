import os
import urllib.request
import zipfile
import glob

def download_and_extract():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
    zip_path = os.path.join(data_dir, "online_retail_ii.zip")
    target_excel = os.path.join(data_dir, "online_retail_II.xlsx")
    
    if os.path.exists(target_excel):
        print(f"Dataset already exists at {target_excel}. Skipping download.")
        return
        
    print("Downloading Online Retail II dataset from UCI...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete. Extracting...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        # The zip extracts as 'online_retail_II.xlsx', but we ensure it is named correctly
        extracted_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
        if extracted_files:
            if extracted_files[0] != target_excel:
                os.rename(extracted_files[0], target_excel)
            print(f"Dataset successfully extracted to {target_excel}")
        else:
            print("Warning: Could not find .xlsx file after extraction.")
            
        # Cleanup the zip file
        os.remove(zip_path)
        
    except Exception as e:
        print(f"Error downloading the dataset: {e}")

if __name__ == "__main__":
    download_and_extract()