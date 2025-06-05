# AMM Risk Lab

This project provides tools for analytics, simulation, and parameter calibration for Automated Market Maker (AMM) liquidity pools, with a focus on Uniswap v3. It includes:

- Fetching and analyzing pool data from The Graph subgraphs
- Simulating LP strategies and computing impermanent loss
- Currency conversion utilities (e.g., token to USD)
- Parameter calibration and autoregressive modeling for pool dynamics
- Price path generation (Geometric Brownian Motion, CPU/GPU)
- Example scripts for testing and displaying results

## Structure

- `src/` - Core C++ source files
- `include/` - C++ headers
- `scripts/` - Example and test scripts (run as executables)
- `utils/` - JSON files and utility data
- `CMakeLists.txt` - Build configuration

## Requirements

- C++17 or later
- [Eigen3](https://eigen.tuxfamily.org/) (for matrix operations)
- [cURL](https://curl.se/libcurl/) (for HTTP requests)
- [nlohmann/json](https://github.com/nlohmann/json) (for JSON parsing)
- CUDA (optional, for GPU price simulation)
- CMake >= 3.10

## Build

```sh
mkdir -p build
cd build
cmake ..
make
```

## Usage

Example: Simulate an LP strategy

```sh
./test_lp_strategy <apiSubgraphs> <idSubgraphs> <poolAddress>
```

Example: Train AR parameters

```sh
./train_parameters <apiSubgraphs> <idSubgraphs> <poolAddress>
```

## Repository

This code is maintained at: [https://github.com/janmarxen/amm-risk-lab](https://github.com/janmarxen/amm-risk-lab)

## License

See [LICENSE](LICENSE) for details.