cmake_minimum_required(VERSION 3.27)
project(MLP_C C)

set(CMAKE_C_STANDARD 11)

add_executable(MLP_C src/main.c
        src/funcionesDeActivacion.c
        src/Layer_Dense.c
        headers/Layer_Dense.h
)
