cmake_minimum_required(VERSION 3.27)
project(rnn_c C)

set(CMAKE_C_STANDARD 11)

add_executable(rnn_c src/main.c
        src/dataReader.c
        src/RNN.c
)
