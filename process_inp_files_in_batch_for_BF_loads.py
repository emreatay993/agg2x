import os
import re

def process_inp_files(input_folder, output_folder_name):
    # Define the regex pattern to match lines starting with "bf,"
    pattern = re.compile(r'^bf,')

    # Ensure the output directory exists
    output_folder = os.path.join(input_folder, output_folder_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.inp'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_name = os.path.splitext(filename)[0] + '_BF_ONLY.inp'
            output_file_path = os.path.join(output_folder, output_file_name)

            # Process each file
            with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
                for line in input_file:
                    if pattern.match(line):
                        output_file.write(line)

            print(f"Processed {filename} and saved output to {output_file_name}")

# Specify the input folder containing .inp files and the output folder name
input_folder = r'path\to\your\input_folder'  # Update this to your actual input folder path

# Process all .inp files in the input folder
process_inp_files(input_folder, "inp_files_BF_ONLY")

print("All files processed successfully.")
