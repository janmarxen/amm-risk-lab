# CMakeLists for PLV_model module

file(GLOB PLV_MODEL_SOURCES "*.cpp")

add_library(PLV_model STATIC ${PLV_MODEL_SOURCES})

target_include_directories(PLV_model PUBLIC
    ${CMAKE_SOURCE_DIR}/include/PLV_model
)
