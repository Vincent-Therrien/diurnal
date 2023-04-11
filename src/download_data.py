import tarfile
import requests

# Repository root URL.
root = "https://raw.githubusercontent.com/Vincent-Therrien/rna-2s-database/main/"

# Individual dataset names.
datasetNames = {
    "archiveII": "archiveII.tar.gz",
}

# Configure the download and formatting file path.
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
output_path = "../data/"

# Download and extract data.
for datasetName, url_ending in datasetNames.items():
    # Download the file.
    url = root + datasetName + "/" + url_ending
    print(url)
    downloaded_file_name = output_path + url_ending
    
    r = requests.get(url, allow_redirects=True)
    open(downloaded_file_name, 'wb').write(r.content)

    # Extract the file.
    print(downloaded_file_name)
    tar = tarfile.open(downloaded_file_name, "r:gz")
    tar.extractall(downloaded_file_name[:-7])
    tar.close()

