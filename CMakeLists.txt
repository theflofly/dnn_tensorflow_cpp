cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DNN_Tensorflow_CPP LANGUAGES CXX)

find_package(tensorflow REQUIRED)

add_executable(main model.cc data_set.cc data_set.h)
target_link_libraries(main PRIVATE tensorflow)

configure_file(normalized_car_features.csv ${CMAKE_CURRENT_BINARY_DIR}/normalized_car_features.csv COPYONLY)
if(MSVC)
    target_compile_definitions(main PRIVATE COMPILER_MSVC)
endif(MSVC)