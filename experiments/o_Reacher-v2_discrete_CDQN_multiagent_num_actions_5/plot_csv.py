
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main(file_prefix):
    # Step 1: Read the contents of the CSV file
    # Replace 'data.csv' with the path to your CSV file
    file_path = f"./{file_prefix}.csv"
    data = pd.read_csv(file_path)

    # Step 2: Plot the data
    colors = ['b','c','g','m','r','y','k']
    plt.figure(figsize=(10, 6))
    for i in range(len(data.columns) -2):
        plt.plot(data['train_step'], data[f'avg_return_{i}'], marker='o', linestyle='-', color=colors[i])
    plt.plot(data['train_step'], data['avg_of_avg_returns'], marker='o', linestyle='-', color=colors[-1])
    plt.title('Average Return vs Train Step')
    plt.xlabel('Train Step')
    plt.ylabel('Average Return')
    plt.grid()
    plt.xticks(rotation=45)
    plt.ylim(-60,0)

    # Step 3: Save the plot to a file
    # You can specify the file format by changing the extension (e.g., .png, .jpg, .pdf)
    output_file_path = f"./{file_prefix}.png"
    plt.savefig(output_file_path)

    # Optionally, show the plot
    # plt.show()

    print(f"Plot saved as {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input text files.")
    parser.add_argument('file_prefix', type=str, help='file name without extension .csv')
    args = parser.parse_args()

    main(args.file_prefix)

