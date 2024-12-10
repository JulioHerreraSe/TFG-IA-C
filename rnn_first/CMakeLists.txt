cmake_minimum_required(VERSION 3.27)
project(rnn_first C)

set(CMAKE_C_STANDARD 11)

add_executable(rnn_first src/main.c
        includes/config.h
        includes/rnn.h
        includes/util.h
        src/rnn.c
        src/util.c)
