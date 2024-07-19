import csv
import json

def preprocess_csv_to_json(input_file, output_file):
    data = []

    with open(input_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        current_record = []
        inside_quotes = False

        for line in infile:
            quote_count = line.count('"')

            if inside_quotes:
                current_record.append(line.strip('\n'))
                if quote_count % 2 != 0:  # Closing quotes found
                    inside_quotes = False
                    data.append(' '.join(current_record))
                    current_record = []
            else:
                if quote_count % 2 != 0:  # Opening quotes found
                    inside_quotes = True
                    current_record.append(line.strip('\n'))
                else:
                    data.append(line.strip('\n'))

        # If there's any remaining content in current_record, add it to data
        if current_record:
            data.append(' '.join(current_record))

    # Split each line into fields based on commas and convert to dictionary
    header = data[0].split(',')
    json_data = []

    for row in data[1:]:
        fields = csv.reader([row])
        for field in fields:
            json_data.append(dict(zip(header, field)))

    # Save the data to a JSON file
    with open(output_file, 'w') as jsonfile:
        json.dump(json_data, jsonfile, indent=4)

input_file = 'sirion_data.csv'
output_file = 'sirion_data.json'

# Preprocess the CSV and save it to JSON
preprocess_csv_to_json(input_file, output_file)

print(f'JSON data has been saved to {output_file}')
