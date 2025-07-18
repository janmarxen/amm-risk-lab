# AMM Risk Lab

This project provides tools for analytics, simulation, and parameter calibration for Automated Market Maker (AMM) liquidity pools (LPs), with a focus on Uniswap v3. **The end goal is a full pipeline that generates synthetic price paths (using Geometric Brownian Motion), predicts liquidity and volume with LSTM and Transformer models, and then computes key risk and return metrics (impermanent loss, fee accumulation, APR, APY, VaR) from these predictions.**

- Fetching and analyzing LP data from The Graph subgraphs
- Simulating LP strategies and computing estimations for impermanent loss, fee accumulation, APR, APY, and VaR (using ML-predicted liquidity/volume)
- Parameter calibration and autoregressive modeling for pool dynamics
- Currency conversion utilities (e.g., token to USD)
- Price path generation (Geometric Brownian Motion, CPU/GPU)
- **ML pipeline for predicting liquidity/volume from prices (LSTM, Transformer)**
- **Future: Reinforcement Learning Liquidity Pool bot for optimal strategy**

## Structure

- `src/` - Core C++ source files
- `include/` - C++ headers
- `scripts/` - Example and test scripts (run as executables)
- `utils/` - utility data
- `CMakeLists.txt` - Build configuration

## Requirements

- C++17 or later
- [Eigen3](https://eigen.tuxfamily.org/) (for matrix operations)
- [cURL](https://curl.se/libcurl/) (for HTTP requests)
- [nlohmann/json](https://github.com/nlohmann/json) (for JSON parsing)
- CUDA (optional, for GPU price simulation)
- CMake >= 3.10

## Installation
_To be added: build instructions, dependencies, CUDA setup_

```sh
mkdir -p build
cd build
cmake ..
make
```

## Usage

_To be added: Create proper folder with examples later_

Example: Simulate an LP strategy

```sh
./test_lp_strategy <apiSubgraphs> <idSubgraphs> <poolAddress>
```

## License

See [LICENSE](LICENSE) for details.

## Contributing
_To be added: guidelines, TODOs, structure_

## Python ML Pipeline (Uniswap V3 Analytics & ML)

This project includes a robust, modular pipeline for Uniswap V3 pool analytics and machine learning, supporting data fetching, feature engineering, storage, PyTorch dataset creation, LSTM model training, grid search, and evaluation. The workflow is fully automated and configurable from a single batch script. In the future, supercomputer support and GPU training will be implemented. 

### Structure
- `python/ml/PLV/scripts/batch_run.sh`: Main batch script to run the full pipeline (data download, training, testing)
- `python/ml/PLV/scripts/run_data_download.py`: Downloads and stores pool data to HDF5
- `python/ml/PLV/scripts/run_training.py`: Trains the LSTM model on multiple pool data
- `python/ml/PLV/scripts/run_testing.py`: Loads, finetunes, and evaluates the model on a specific pool. Each pool should have its own finetuned model.
- `python/ml/PLV/data_io.py`: Feature engineering, dataset utilities
- `python/ml/PLV/model.py`: ZeroInflatedLSTM model, custom loss, scaling, early stopping

### Requirements
- Python 3.8+
- PyTorch, pandas, numpy, scikit-learn, matplotlib, tables (PyTables)

### Usage

1. **Configure the pipeline**: Edit `python/ml/PLV/scripts/batch_run.sh` to set all parameters at the top (API key, subgraph ID, date ranges, model hyperparameters, number of pools, etc).

2. **Run the pipeline**:
   ```sh
   ./python/ml/PLV/scripts/batch_run.sh
   ```
   This will:
   - Download pool data (with `N_POOLS` limit and always including the main pool)
   - Train the LSTM (or Transformer) model with early stopping (using validation split)
   - Finetune and evaluate the model on the main pool
