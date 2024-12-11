import os
import pandas as pd
from PyQt5.QtWidgets import QApplication, QFileDialog


def convert_txt_to_apdl_in_folder():
    # Initialize PyQt5 application
    app = QApplication([])
    folder_path = QFileDialog.getExistingDirectory(None, "Select Folder Containing TXT Files")
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    # Create a new subfolder for the output files
    output_folder = os.path.join(folder_path, "APDL_Files")
    os.makedirs(output_folder, exist_ok=True)

    # Process all .txt files in the selected folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(folder_path, filename)
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_apdl.inp")

            try:
                # Read the TXT file using tab delimiter
                df = pd.read_csv(input_file_path, delimiter="\t", encoding="ISO-8859-1")

                # Identify Node Number and Temperature columns
                node_col = [col for col in df.columns if "Node" in col or "Number" in col][0]
                temp_col = [col for col in df.columns if "Temperature" in col or "Temp" in col][0]

                # Check for NaN values
                if df[node_col].isnull().any() or df[temp_col].isnull().any():
                    print(f"Skipping {filename}: Contains NaN values in Node or Temperature columns.")
                    continue

                # Generate APDL commands
                with open(output_file_path, "w") as f:
                    f.write("/prep7\n")  # Begin preprocessor
                    for _, row in df.iterrows():
                        node_number = int(row[node_col])
                        temperature = row[temp_col]
                        f.write(f"bf,{node_number},temp,{temperature}\n")
                    f.write("/solu\n")  # Transition to the solution processor

                print(f"Processed: {filename} -> {output_file_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"All files processed. APDL files are saved in: {output_folder}")


if __name__ == "__main__":
    convert_txt_to_apdl_in_folder()
