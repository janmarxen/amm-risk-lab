import os
import json
import glob

RESULTS_DIR = "/p/project1/training2529/marxen1/amm-risk-lab/python/ml/PLV/models/gs_results"

def main():
    """
    Aggregates all JSON result files in the grid search results directory and prints the parameters
    of the model with the lowest validation loss.
    Searches for files in RESULTS_DIR, loads each JSON, and compares 'val_loss' and 'params'.
    Prints the best model's file, validation loss, and parameters. If no valid results are found,
    prints a message.
    """
    result_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))
    if not result_files:
        print("No result files found in gs_results.")
        return
    best_loss = float('inf')
    best_params = None
    best_file = None
    for file in result_files:
        with open(file, 'r') as f:
            data = json.load(f)
            val_loss = data.get('val_loss', None)
            params = data.get('params', None)
            if val_loss is not None and params is not None:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
                    best_file = file
    if best_params is not None:
        print(f"Best model found in: {best_file}")
        print(f"Validation loss: {best_loss}")
        print("Best model parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    else:
        print("No valid results found.")

if __name__ == "__main__":
    main()
