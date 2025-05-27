import os
import shutil

# Path to the folder containing the video sequences
source_folder = 'datasets/Vimeo90K/raw/test'

# Traverse the files and subfolders
for root, dirs, files in os.walk(source_folder):
    for file in files:
        # Check if the file is an image (e.g., .png)
        if file.endswith('.png'):
            # Get the full path of the file
            file_path = os.path.join(root, file)

            # Extract the parts of the directory path (e.g., id1, bla1)
            rel_path = os.path.relpath(file_path, source_folder)
            parts = rel_path.split(os.sep)  # Split the path based on /

            if len(parts) >= 2:
                # Create a new folder name by combining the first two parts (e.g., id1_bla1)
                new_folder_name = f"{parts[0]}_{parts[1]}"
                new_folder_path = os.path.join(source_folder, new_folder_name)

                # Create the new folder if it doesn't exist
                os.makedirs(new_folder_path, exist_ok=True)

                # Move the file to the new folder
                shutil.move(file_path, os.path.join(new_folder_path, file))
                print(f"Moved {file_path} to {os.path.join(new_folder_path, file)}")


# Now, remove the old empty directories
for root, dirs, files in os.walk(source_folder, topdown=False):
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)

        # Check if the directory is empty and is a parent directory (has no subdirectories or files)
        if not os.listdir(dir_path):  # If the directory is empty
            # Remove the empty parent directory
            os.rmdir(dir_path)
            print(f"Removed empty directory: {dir_path}")