import os
import shutil
import zipfile
from google.colab import files
from google.colab import drive

def save_history(action='download'):
    # Ensure the 'history' directory exists
    if not os.path.exists('history'):
        print("The 'history' directory does not exist.")
        return

    # Zip the 'history' directory
    shutil.make_archive('history', 'zip', 'history')
    print("Directory 'history' has been zipped successfully.")

    # Action to download the zip file to local storage
    if action == 'download':
        files.download('history.zip')
        print("File 'history.zip' is being downloaded to your local storage...")

    # Action to upload the zip file to Google Drive
    elif action == 'drive':
        # Mount Google Drive
        drive.mount('/content/drive')
        # Specify the path in Google Drive to save the zip file
        drive_path = '/content/drive/My Drive/history.zip'
        # Copy the file to Google Drive
        shutil.copy('history.zip', drive_path)
        print(f"File 'history.zip' has been uploaded to Google Drive at: {drive_path}")

def upload_history():
    # Upload the zip file
    uploaded = files.upload()
    zip_file_name = None

    # Find the uploaded zip file name
    for fn in uploaded.keys():
        if fn.endswith('.zip'):
            zip_file_name = fn
            break

    # If a zip file was uploaded, unzip it to 'history'
    if zip_file_name:
        # Make sure any existing 'history' directory is removed first
        if os.path.exists('history'):
            shutil.rmtree('history')
        
        # Unzip the file
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall('history')
        print(f"'{zip_file_name}' has been unzipped into the 'history' directory.")
        
        # Optionally, clean up the uploaded zip file after extraction
        os.remove(zip_file_name)
    else:
        print("No zip file found in the uploaded files.")
