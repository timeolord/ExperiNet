cmake_minimum_required(VERSION 3.16)
project(StockTrader)

set(CMAKE_CXX_STANDARD 20)

add_executable(StockTrader main.cpp Layer.h ActivationFunctions.h Network.h CostFunctions.h)