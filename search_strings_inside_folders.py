import os
import fnmatch
import tkinter as tk
from tkinter import filedialog, scrolledtext

def search_in_files(folder_path, search_strings, output_area, current_folder_var):
    if not isinstance(search_strings, list):
        search_strings = [search_strings]  # Ensure search_strings is a list for consistency

    for dirpath, dirnames, filenames in os.walk(folder_path):
        current_folder_var.set(f"Currently searching in: {dirpath}")  # Update the current directory being searched
        output_area.update()
        
        for filename in filenames:
            # Search in file names
            if any(search_string in filename for search_string in search_strings):
                output_area.insert(tk.END, f"String found in file name: {os.path.join(dirpath, filename)}\n")

            # Further processing only for .py files
            if fnmatch.fnmatch(filename, '*.py'):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    line_num = 1  # Initialize line number
                    for line in file:
                        # Search inside file content
                        if any(search_string in line for search_string in search_strings):
                            output_area.insert(tk.END, f"String found in file content: {file_path}, Line: {line_num}\n")
                        line_num += 1

def start_search():
    folder_path = folder_path_entry.get()
    search_strings = search_strings_entry.get().split(',')
    search_strings = [s.strip() for s in search_strings]
    result_area.delete(1.0, tk.END)  # Clear the result area before new search
    search_in_files(folder_path, search_strings, result_area, current_folder_var)

def browse_folder():
    folder_path = filedialog.askdirectory()
    folder_path_entry.delete(0, tk.END)
    folder_path_entry.insert(0, folder_path)

# Set up the GUI
root = tk.Tk()
root.title("Search in Files")
root.geometry("800x600")  # Initial size of the window

# Make the GUI resizable
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_rowconfigure(4, weight=1)
root.grid_rowconfigure(5, weight=10)

frame = tk.Frame(root)
frame.grid(sticky='nsew')

# Folder path input
folder_path_label = tk.Label(frame, text="Folder Path:")
folder_path_label.grid(row=0, column=0, sticky='w')
folder_path_entry = tk.Entry(frame, width=50)
folder_path_entry.grid(row=1, column=0, sticky='ew')
browse_button = tk.Button(frame, text="Browse", command=browse_folder)
browse_button.grid(row=1, column=1, sticky='ew')

# Search strings input
search_strings_label = tk.Label(frame, text="Search Strings (comma-separated):")
search_strings_label.grid(row=2, column=0, sticky='w')
search_strings_entry = tk.Entry(frame, width=50)
search_strings_entry.grid(row=3, column=0, sticky='ew')

# Search button
search_button = tk.Button(frame, text="Search", command=start_search)
search_button.grid(row=3, column=1, sticky='ew')

# Display current folder being searched
current_folder_var = tk.StringVar()
current_folder_label = tk.Label(frame, textvariable=current_folder_var)
current_folder_label.grid(row=4, column=0, sticky='ew', columnspan=2)

# Result area
result_area = scrolledtext.ScrolledText(frame, height=15)
result_area.grid(row=5, column=0, sticky='nsew', columnspan=2)

frame.grid_columnconfigure(0, weight=1)
frame.grid_rowconfigure(5, weight=1)

root.mainloop()
