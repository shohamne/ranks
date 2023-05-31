import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator

def visualize_metrics(test_name):
    pdf_file = f"{test_name}.pdf"

    # Read the CSV file
    n_tests = 0
    df = 0
    for csv_file in glob.glob(f'{test_name}*.csv'):    
        df += pd.read_csv(csv_file)
        n_tests += 1
    df = df/n_tests
    df.to_csv(f'{test_name}.csv', index=False)

    df = df[df['Epoch'] >= 1]
    # Extract the required metrics data
    layer_indices = df['Layer'].unique()
    depth = df['Depth'].unique()[0]
    metrics = ['Smooth Rank (λ=1.0)', 'Train Loss', 'Test Loss']

    # Create separate subplots for each metric
    num_metrics = len(metrics)
    fig, axs = plt.subplots(num_metrics-1, 1, figsize=(10, 6), sharex=True)

    # Plot graphs for each metric
    for i, metric in enumerate(metrics):
        is_loss = metric in  ('Train Loss', 'Test Loss')
        ax = axs[i] if not is_loss else axs[-1]
        ax.set_ylabel(metric)
        ax.set_title(f'test_name: {test_name}, n_tests: {n_tests}, {metric} over Iterations (Depth: {depth})')
        for layer_index in layer_indices if not is_loss else[0]:
            if metric == ('Train Loss' or metric == 'Test Loss') and layer_index > 0:
                continue
            layer_df = df[df['Layer'] == layer_index]
            metric_values = layer_df[metric]
            epoch = layer_df['Epoch']
            ax.plot(epoch, metric_values, label=f'Layer {int(layer_index)}')

        ax.legend()

        # Set y-axis to Symmetrical Log Scale
        ax.set_yscale('symlog', linthresh=0.1, base=10)
        ax.yaxis.set_major_locator(SymmetricalLogLocator(linthresh=0.1, base=10))

        # Set x-axis to Log Scale
        # ax.set_xscale('log')

    # Set x-label for the last subplot
    axs[-1].set_xlabel('Global Iteration')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot to PDF
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize metrics from a CSV file")
    parser.add_argument("--test_name", default="metrics", type=str, help="Test name")
    args = parser.parse_args()

    # Visualize the metrics
    visualize_metrics(args.test_name)
