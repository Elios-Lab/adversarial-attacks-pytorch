import os
import shutil

def copy_and_rename_files(source_dir, target_dir, append_str):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Loop through all files in the source directory
    for filename in os.listdir(source_dir):
        # Construct the full file path
        source_file = os.path.join(source_dir, filename)
        
        # Check if it is a file (not a directory)
        if os.path.isfile(source_file):
            # Split the filename and its extension
            name, ext = os.path.splitext(filename)
            
            # Create the new filename by appending the user-defined string
            new_filename = f"{name}{append_str}{ext}"
            
            # Construct the full target file path
            target_file = os.path.join(target_dir, new_filename)
            
            # Copy the file to the target directory with the new name
            shutil.copy2(source_file, target_file)

    print("Files copied and renamed successfully.")

# Define the source and target directories and the string to append
source_dir = "/home/pigo/Desktop/BoschDriveU5px/test/5px/labels/"
target_dir = "/home/pigo/Desktop/BoschDriveU5px/pixle_test_50res_25miter_True/labels/"
append_str = "_Pixle"

# Call the function
copy_and_rename_files(source_dir, target_dir, append_str)