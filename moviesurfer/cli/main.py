import os
import wget
import zipfile

# Function to download data
def download_data():
    output_dir = './moviesurfer/data'
    url = 'http://files.grouplens.org/datasets/movielens/ml-25m.zip'
    filename = wget.download(url, out=output_dir)
    
    # extract data using zipfile
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        
    # delete zip file
    os.remove(filename)
    
def main():
    print("Hello from CLI, functionality is to be added")