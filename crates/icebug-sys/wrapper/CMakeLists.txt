cmake_minimum_required(VERSION 3.14)
project(icebug_wrapper CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build NetworKit (icebug) as a static library
set(NETWORKIT_BUILD_CORE ON CACHE BOOL "" FORCE)
set(NETWORKIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NETWORKIT_MONOLITH ON CACHE BOOL "" FORCE)
set(NETWORKIT_STATIC ON CACHE BOOL "" FORCE)

# Add icebug source
add_subdirectory(${ICEBUG_SOURCE_DIR} ${CMAKE_BINARY_DIR}/icebug)

# Build our C wrapper as a static library
add_library(icebug_c STATIC icebug_c.cpp)
target_include_directories(icebug_c PRIVATE
    ${ICEBUG_SOURCE_DIR}/include
    ${ICEBUG_SOURCE_DIR}/extlibs
)
target_link_libraries(icebug_c PRIVATE networkit)

# Install targets
install(TARGETS icebug_c networkit
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
