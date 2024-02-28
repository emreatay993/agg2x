import re

def get_lines_starting_with_bf_and_write_to_file(input_file_path, output_file_path):
    pattern = re.compile(r'^bf,')  # Compile the regex pattern once for efficiency
    desired_lines = []

    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            if pattern.match(line):  # Use the compiled pattern for matching
                desired_lines.append(line)  # Keep the newline character for writing to file

    # Write the filtered lines to the output file
    with open(output_file_path, 'w') as output_file:
        output_file.writelines(desired_lines)

# Specify the paths to your input and output files
input_file_path = '/path/to/your/input_file.dat'  # Update this to your actual input file path
output_file_path = '/path/to/your/output_file.dat'  # Update this to your desired output file path

# Call the function with the input and output file paths
get_lines_starting_with_bf_and_write_to_file(input_file_path, output_file_path)

print(f"Filtered lines have been written to: {output_file_path}")
