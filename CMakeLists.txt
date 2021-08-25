cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(StockTrader)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent) # If not included already

FetchContent_Declare(sciplot_content
        GIT_REPOSITORY https://github.com/sciplot/sciplot.git
        GIT_TAG master)

FetchContent_GetProperties(sciplot_content)
if(NOT sciplot_content_POPULATED)
    FetchContent_Populate(sciplot_content)
endif()

include_directories(${sciplot_content_SOURCE_DIR})

add_executable(StockTrader main.cpp Layer.h ActivationFunctions.h Network.h CostFunctions.h)