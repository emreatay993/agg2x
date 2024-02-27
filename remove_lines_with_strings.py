def remove_lines_with_strings(input_file, output_file, strings_to_remove):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Check if any unwanted string is in the current line
            if not any(unwanted in line for unwanted in strings_to_remove):
                outfile.write(line)

# Specify your input and output file paths
input_file_path = 'input.txt'  # Change this to the path of your input file
output_file_path = 'output.txt'  # Change this to the desired path for your output file

# Strings to look for in each line
strings_to_remove = ['UMID', 'UVID']

# Call the function to remove lines containing specified strings
remove_lines_with_strings(input_file_path, output_file_path, strings_to_remove)
