import pandas as pd
import argparse
import glob

def add_column(data, file_name):
    with open(file_name, 'r') as file:
        for line in file:
            # Checking if the line contains 'train_step' and 'avg_return' 
            if 'train_step' in line and 'avg_return' in line:
                # Extracting the train_step and avg_return values
                parts = line.split()
                train_step = None
                avg_return = None
                for part in parts:
                    if 'train_step' in part:
                        train_step = int(part.split('=')[1])
                    elif 'avg_return' in part:
                        avg_return = float(part.split('=')[1])
                
                if train_step is not None and avg_return is not None:
                    # if train_step not in data:
                    #     data[train_step] = [None] * k  # Reserve space for avg_returns
                    # idx = len([x for x in data[train_step] if x is not None])  # Get current index
                    if str(train_step) in data:
                        data[str(train_step)] = data[str(train_step)] + [avg_return]
                    else:
                        data[str(train_step)] = [avg_return]

def save_to_csv(data, output_file):
    # Create a DataFrame from the gathered data
    rows = []
    for train_step, avg_returns in data.items():
        avg_return_value = sum([x for x in avg_returns if x is not None]) / len([x for x in avg_returns if x is not None])
        row = [int(train_step)] + avg_returns + [avg_return_value]
        rows.append(row)
    
    # Create a DataFrame and save it to a CSV file
    columns = ['train_step'] + [f'avg_return_{i + 1}' for i in range(len(avg_returns))] + ['avg_of_avg_returns']
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")

def main(k, file_prefix):
    # file_prefix = 'o_tmp'  # Change this to the base name of your files
    data = {}  # dictionary
    
    # Add columns for each file
    for i in range(k):
        file_name = f"{file_prefix}_{i}"  # Assuming the file extensions are .txt
        add_column(data, file_name)
    
    # Save the extracted data to CSV
    output_file = f"{file_prefix}.csv"
    save_to_csv(data, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input text files.")
    parser.add_argument('k', type=int, help='Number of input files (excluding the base name)')
    parser.add_argument('file_prefix', type=str, help='file prefix (file name would be file_prefix_k)')
    args = parser.parse_args()
    
    main(args.k, args.file_prefix)

