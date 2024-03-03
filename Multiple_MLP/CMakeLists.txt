cmake_minimum_required(VERSION 3.27)
project(Multiple_MLP C)

set(CMAKE_C_STANDARD 11)

add_executable(Multiple_MLP src/main.c
        src/mlp.c
        includes/utils.h
        src/utils.c
)
