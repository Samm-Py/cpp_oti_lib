cmake_minimum_required(VERSION 3.18)

project(otinum_consumer LANGUAGES CXX)

find_package(otinum CONFIG REQUIRED)

add_executable(otinum_consumer main.cpp)
target_link_libraries(otinum_consumer PRIVATE oti::otinum)

enable_testing()
add_test(NAME otinum_consumer COMMAND otinum_consumer)
