import pandas as pd

def generate_cmblock_from_csv(csv_path, component_name="P_ATM_NODES", column_name="NodeNumber"):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get the NodeNumber column as integers
    node_numbers = df[column_name].dropna().astype(int).tolist()
    
    # Count the number of nodes
    node_count = len(node_numbers)

    # Header line of the CMBLOCK command
    output_lines = [f"CMBLOCK,{component_name},NODE,{node_count}"]
    
    # APDL format specifier line
    output_lines.append("(8I10)")
    
    # Format node numbers into 8 per line, 10-character wide integers
    for i in range(0, node_count, 8):
        chunk = node_numbers[i:i+8]
        formatted_line = ''.join(f"{num:10d}" for num in chunk)
        output_lines.append(formatted_line)
    
    return '\n'.join(output_lines)

# Example usage
csv_file = "your_file.csv"  # Replace with your actual path
apdl_block = generate_cmblock_from_csv(csv_file)

# Print or write to a file
print(apdl_block)

# To save to a file, uncomment:
# with open("cmblock_output.txt", "w") as f:
#     f.write(apdl_block)
