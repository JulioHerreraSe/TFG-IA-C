cmake_minimum_required(VERSION 3.27)
project(rnn_c C)

set(CMAKE_C_STANDARD 11)

include_directories(src)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(lib/googletest)