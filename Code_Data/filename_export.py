import os
import csv

def save_filenames_to_csv(folder_path, output_csv):
    # Get the list of files in the specified folder
    files = os.listdir(folder_path)

    # Create and write to the CSV file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['File Name'])

        # Write the file names
        for filename in files:
            writer.writerow([filename])

# Specify the folder path and output CSV file name
folder_path = 'Data\RAW'  # Replace with your folder path
output_csv = 'Data\RAW\Jaipur_Filenames.csv'          # Output CSV file name

# Call the function
save_filenames_to_csv(folder_path, output_csv)

print(f"File names saved to {output_csv}")
