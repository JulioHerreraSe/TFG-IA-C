cmake_minimum_required(VERSION 3.27)
project(MLP_Update C)

set(CMAKE_C_STANDARD 11)

add_executable(MLP_Update src/main.c
        src/dataReader.c
        includes/dataReader.h
        src/MLP.c
        includes/MLP.h
        includes/config.h)
