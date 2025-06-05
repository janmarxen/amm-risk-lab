# AMM Risk Lab

This project provides tools for analytics, simulation, and parameter calibration for Automated Market Maker (AMM) liquidity pools (LPs), with a focus on Uniswap v3. It includes:

- Fetching and analyzing LP data from The Graph subgraphs
- Simulating LP strategies and computing estimations for impermanent loss, fee accumulation, APR, APY, and VaR estimation
- Parameter calibration and autoregressive modeling for pool dynamics
- Currency conversion utilities (e.g., token to USD)
- Price path generation (Geometric Brownian Motion, CPU/GPU)

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