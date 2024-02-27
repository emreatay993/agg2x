# Execute the updated function to drop lines containing "MP,UVID" or "MP,UMID"

def reformat_bfblock_for_temperature_data(input_file_path, output_file_path):
    bfblock_start_pattern = re.compile(r'^bfblock,')
    bfblock_end_pattern = re.compile(r'^bf,end')
    temp_data_pattern = re.compile(r'^\s*(\d+)\s+([\deE.+-]+)')
    skip_patterns = [re.compile(r'MP,UVID'), re.compile(r'MP,UMID')]

    output_lines = []
    temp_data_started = False

    with open(input_file_path, 'r') as file:
        for line in file:
            # Check if the line should be skipped due to containing specific strings
            if any(pattern.search(line) for pattern in skip_patterns):
                continue  # Skip lines containing "MP,UVID" or "MP,UMID"

            # Remove trailing newline character for processing
            line_stripped = line.rstrip('\n')

            if bfblock_start_pattern.match(line_stripped):
                temp_data_started = True
                continue  # Skip the bfblock line and the following line

            if bfblock_end_pattern.match(line_stripped):
                temp_data_started = False
                continue  # Skip the bf,end line

            if temp_data_started:
                match = temp_data_pattern.match(line_stripped)
                if match:
                    index, value = match.groups()
                    line_stripped = f"bf,{index},temp,{value}"
                else:
                    continue  # Skip non-data lines within the temperature block

            # Retain the original line ending
            output_lines.append(line_stripped + '\n')

    # Write the processed lines to the output file, avoiding an extra newline at the end
    with open(output_file_path, 'w') as output_file:
        output_file.writelines(output_lines[:-1] + [output_lines[-1].rstrip('\n')])

# Updated file paths for the function that drops specific lines
input_file_path = '/mnt/data/ds.dat'
output_file_path = '/mnt/data/ds_processed_dropped_lines.dat'

# Run the updated function
reformat_bfblock_for_temperature_data(input_file_path, output_file_path)

output_file_path
