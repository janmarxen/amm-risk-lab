cmake_minimum_required(VERSION 3.10)
project(liquidity_pools_analytics)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(CURL REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CURL_INCLUDE_DIRS})
include_directories(src)
include_directories(include)

# Add utils directory
include_directories(include/utils)

# Add lp_strategy.cpp to your sources
set(SOURCES
    src/common.cpp
    src/lp_strategy.cpp
    src/utils/conversions.cpp
    src/utils/subgraph_utils.cpp
    # ...other source files...
)

# add_executable(liquidity_pools_analytics
#     src/price_generation.cpp
#     src/common.cpp
#     src/model_training.cpp
#     src/pool_data_fetching.cpp
# )

# target_link_libraries(liquidity_pools_analytics
#     Eigen3::Eigen
#     CURL::libcurl
# )

add_executable(display_pool_data
    scripts/display_pool_data.cpp
    src/pool_data_fetching.cpp
    src/utils/time_utils.cpp
    src/utils/subgraph_utils.cpp
)
target_link_libraries(display_pool_data
    CURL::libcurl
)

add_executable(test_lp_strategy 
    scripts/test_lp_strategy.cpp 
    src/lp_strategy.cpp 
    src/utils/subgraph_utils.cpp
    src/pool_data_fetching.cpp 
    src/common.cpp 
    src/utils/time_utils.cpp
    src/utils/conversions.cpp
)
target_include_directories(test_lp_strategy PRIVATE include)
target_link_libraries(test_lp_strategy
    CURL::libcurl
)

add_executable(test_conversions
    scripts/test_conversions.cpp
    src/pool_data_fetching.cpp
    src/utils/conversions.cpp
    src/utils/subgraph_utils.cpp
    src/utils/time_utils.cpp
    src/common.cpp
)
target_include_directories(test_conversions PRIVATE include)
target_link_libraries(test_conversions
    CURL::libcurl
)

add_executable(test_price_generation
    scripts/test_price_generation.cpp
    src/price_generation.cpp
)
target_include_directories(test_price_generation PRIVATE include)
target_link_libraries(test_price_generation
    Eigen3::Eigen
)

