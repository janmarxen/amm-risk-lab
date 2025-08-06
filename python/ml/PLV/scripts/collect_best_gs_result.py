import os
import json

RESULTS_DIR = "/p/project1/training2529/marxen1/amm-risk-lab/python/ml/PLV/models"

def main():
    """
    Aggregates all results from the single JSONL file and prints the parameters
    of the model with the lowest validation loss.
    """
    # Look for the single results file
    result_files = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith('_gridsearch_results.jsonl'):
            result_files.append(os.path.join(RESULTS_DIR, filename))
    
    if not result_files:
        print("No grid search results file found.")
        print(f"Looking for files ending with '_gridsearch_results.jsonl' in {RESULTS_DIR}")
        return
    
    best_loss = float('inf')
    best_params = None
    best_combination_id = None
    total_results = 0
    
    for results_file in result_files:
        print(f"Reading results from: {results_file}")
        
        try:
            with open(results_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        data = json.loads(line)
                        total_results += 1
                        
                        val_loss = data.get('val_loss', None)
                        params = data.get('params', None)
                        combination_id = data.get('combination_id', line_num)
                        
                        if val_loss is not None and params is not None:
                            if val_loss < best_loss:
                                best_loss = val_loss
                                best_params = params
                                best_combination_id = combination_id
                    
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line {line_num} in {results_file}: {e}")
                        continue
        
        except FileNotFoundError:
            print(f"Results file not found: {results_file}")
            continue
    
    print(f"\nProcessed {total_results} grid search results.")
    
    if best_params is not None:
        print(f"Best model found: Combination ID {best_combination_id}")
        print(f"Validation loss: {best_loss:.8f}")
        print("Best model parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    else:
        print("No valid results found.")

if __name__ == "__main__":
    main()
