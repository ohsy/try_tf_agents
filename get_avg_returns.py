import pandas as pd
import argparse
import glob

def add_column(data, file_name):
    with open(file_name, 'r') as file:
        for line in file:
            # Checking if the line contains 'time_step' and 'avg_return' 
            if 'time_step' in line and 'avg_return' in line:
                # Extracting the time_step and avg_return values
                parts = line.split()
                time_step = None
                avg_return = None
                for part in parts:
                    if 'time_step' in part:
                        time_step = int(part.split('=')[1])
                    elif 'avg_return' in part:
                        avg_return = float(part.split('=')[1])
                
                if time_step is not None and avg_return is not None:
                    # if time_step not in data:
                    #     data[time_step] = [None] * k  # Reserve space for avg_returns
                    # idx = len([x for x in data[time_step] if x is not None])  # Get current index
                    if str(time_step) in data:
                        data[str(time_step)] = data[str(time_step)] + [avg_return]
                    else:
                        data[str(time_step)] = [avg_return]


def save_to_csv(data, output_file):
    # Create a DataFrame from the gathered data
    rows = []
    avg_returns_len = 0
    for time_step, avg_returns in data.items():
        avg_returns_len = len(avg_returns)
        avg_return_value = sum([x for x in avg_returns if x is not None]) / len([x for x in avg_returns if x is not None])
        row = [int(time_step)] + avg_returns + [avg_return_value]
        rows.append(row)
    
    # Create a DataFrame and save it to a CSV file
    columns = ['time_step'] + [f'avg_return_{i}' for i in range(avg_returns_len)] + ['avg_of_avg_returns']
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

