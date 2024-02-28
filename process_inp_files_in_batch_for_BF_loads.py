import clr
import os
import re

# Add references to access Windows Forms components
clr.AddReference('System.Windows.Forms')
from System.Windows.Forms import FolderBrowserDialog, DialogResult

def select_folder_dialog():
    dialog = FolderBrowserDialog()
    if dialog.ShowDialog() == DialogResult.OK:
        return dialog.SelectedPath
    else:
        return None

def process_inp_files(input_folder, output_folder_name):
    pattern = re.compile(r'^bf,')
    output_folder = os.path.join(input_folder, output_folder_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.inp'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_name = os.path.splitext(filename)[0] + '_BF_ONLY.inp'
            output_file_path = os.path.join(output_folder, output_file_name)

            with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
                for line in input_file:
                    if pattern.match(line):
                        output_file.write(line)

            print("Processed {0} and saved output to {1}".format(filename, output_file_name))

# Use the dialog to select the input folder
input_folder = select_folder_dialog()

if input_folder:
    print("Selected folder: " + input_folder)
    process_inp_files(input_folder, "inp_files_BF_ONLY")
    print("All files processed successfully.")
else:
    print("No folder selected.")
