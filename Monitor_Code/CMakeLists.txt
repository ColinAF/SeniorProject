# The following lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.5)

set(EXTRA_COMPONENT_DIRS "$ENV{IDF_PATH}/frameworks/esp-idf-v4.4/components/"
"C:/Espressif/frameworks/esp-idf-v4.4/esp-who/components"
"$ENV{IDF_PATH}/examples/common_components/")

add_compile_options(-fdiagnostics-color=always)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(Monitor_Code)
