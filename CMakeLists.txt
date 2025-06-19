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

# Add PLV_model source and include directories
add_subdirectory(src/PLV_model)

include_directories(
    ${CMAKE_SOURCE_DIR}/include/PLV_model
)

# Add lp_strategy.cpp to your sources
set(SOURCES
    src/common.cpp
    src/lp_simulation.cpp
    src/utils/conversions.cpp
    src/utils/subgraph_utils.cpp
    # ...other source files...
)

# Add utils sources as a static library so it is built and linked where needed
file(GLOB UTILS_SOURCES
    src/utils/*.cpp
)
add_library(utils STATIC ${UTILS_SOURCES})
target_include_directories(utils PUBLIC ${CMAKE_SOURCE_DIR}/include/utils)

# Copy the utils folder to the build directory after build
add_custom_command(
    TARGET utils POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/utils
        ${CMAKE_BINARY_DIR}/utils
)

add_executable(display_pool_data
    scripts/display_pool_data.cpp
    src/pool_data_fetching.cpp
    src/utils/time_utils.cpp
    src/utils/subgraph_utils.cpp
)
target_link_libraries(display_pool_data
    CURL::libcurl
    utils
)

add_executable(test_lp_strategy 
    scripts/test_lp_strategy.cpp 
    src/lp_simulation.cpp 
    src/utils/subgraph_utils.cpp
    src/pool_data_fetching.cpp 
    src/common.cpp 
    src/utils/time_utils.cpp
    src/utils/conversions.cpp
)
target_include_directories(test_lp_strategy PRIVATE include)
target_link_libraries(test_lp_strategy
    CURL::libcurl
    utils
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
    utils
)

add_executable(test_price_generation
    scripts/test_price_generation.cpp
    src/price_generation.cpp
    src/utils/time_utils.cpp
    src/common.cpp
)
target_include_directories(test_price_generation PRIVATE include)
target_link_libraries(test_price_generation
    Eigen3::Eigen
    utils
)

add_executable(test_prediction
    scripts/test_prediction.cpp
    src/utils/subgraph_utils.cpp
    src/pool_data_fetching.cpp 
    src/common.cpp 
    src/utils/time_utils.cpp
    src/utils/conversions.cpp
    src/price_generation.cpp
    src/PLV_model/prediction.cpp
    src/lp_simulation.cpp 
    src/monte_carlo.cpp
)
target_include_directories(test_prediction PRIVATE include)
target_link_libraries(test_prediction
    Eigen3::Eigen
    CURL::libcurl
    utils
)


