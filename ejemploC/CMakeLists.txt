cmake_minimum_required(VERSION 3.27)
project(ejemploC C)

set(CMAKE_C_STANDARD 11)

add_executable(ejemploC src/main.c
        src/dataReader.c
        includes/dataReader.h)
