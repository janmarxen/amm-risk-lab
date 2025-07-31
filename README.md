# AMM Risk Lab

This project provides tools for analytics, simulation, and automated rebalancing for Automated Market Maker (AMM) liquidity pools (LPs), with a focus on Uniswap v3. The end goals are:
 1. a full pipeline that generates synthetic price paths (using Geometric Brownian Motion), predicts liquidity and volume with LSTM and Transformer models, and then computes key risk and return metrics (impermanent loss, fee accumulation, APR, APY, VaR) from these predictions.
 2. a fully autonomous RL bot for automated LP position rebalancing which uses simulation for decision making. 
 
Features include:
- Fetching and analyzing LP data from The Graph's Subgraphs
- Simulating LP strategies and computing estimations for impermanent loss, fee accumulation, APR, APY, and VaR (using ML-predicted liquidity/volume)
- Parameter calibration and autoregressive modeling for pool dynamics
- Real-time LP token conversion utilities (e.g., token to USD)
- Price path generation (Geometric Brownian Motion, with GPU support)
- ML pipeline for predicting liquidity/volume from prices (LSTM, Transformer)
- Future: Reinforcement Learning Liquidity Pool bot for optimal strategy

## Structure

- `src/` - Core C++ source files
- `include/` - C++ headers
- `scripts/` - Example and test scripts (run as executables)
- `utils/` - utility data
- `python/ml/PLV` - Price-Liquidity-Volume models for predicting liquidity and volume from prices. Used for Monte-Carlo simulations in `src/`.  
- `CMakeLists.txt` - Build configuration

## Requirements
See `$HOME$/course/sc_venv_template/requirements.txt` and 
[Supercomputing Environment Template](https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template/-/tree/master?ref_type=heads).

## Installation
_To be added: build instructions, dependencies, CUDA setup_

```sh
mkdir -p build
cd build
cmake ..
make
```

## License

See [LICENSE](LICENSE) for details.

## Contributing
_To be added: guidelines, TODOs, structure_

## Python ML Pipeline (Uniswap V3 Analytics & ML)

This project includes a modular, distributed pipeline for Uniswap V3 pool analytics and machine learning. It supports data fetching, feature engineering, HDF5 storage, PyTorch dataset creation, distributed (DDP) model pretraining, grid search, finetuning, and evaluation for both LSTM and Transformer models.

### Structure (Distributed Workflow)
- `python/ml/PLV/scripts/batch_run_data_download.sh`: Batch script to run parallel data download for multiple pools
- `python/ml/PLV/scripts/run_data_download.py`: Downloads and stores pool data to HDF5
- `python/ml/PLV/scripts/run_ddp_gridsearch.py`: Distributed grid search for hyperparameter tuning
- `python/ml/PLV/scripts/run_ddp_pretraining.py`: Distributed (DDP) pretraining of the model on multiple pools
- `python/ml/PLV/scripts/run_ddp_finetuning.py`: Distributed finetuning of the pretrained model on a specific pool
- `python/ml/PLV/scripts/run_testing.py`: Evaluation/testing of the finetuned model
- `python/ml/PLV/data_io.py`: Feature engineering, dataset utilities
- `python/ml/PLV/model.py`: Model definitions (ZeroInflatedLSTM, Transformer), custom loss, early stopping
- `python/ml/PLV/utils/distributed_utils.py`: Utilities for distributed training, model/scaler saving/loading
