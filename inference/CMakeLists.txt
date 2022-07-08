# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.12)

project(EncryptedFeatureFusion VERSION 1.0 LANGUAGES CXX)

# Executable will be in ../../bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../bin")

# Import Microsoft SEAL
find_package(SEAL 4.0.0 EXACT REQUIRED)

add_executable(encrypted_feature_fusion encrypted_feature_fusion.cpp)

if(SEAL_FOUND)
    message("SEAL Found")
    include_directories(${SEAL_INCLUDE_DIRS}, "../../include/")
    target_link_libraries(encrypted_feature_fusion SEAL::seal_shared)
elseif(NOT SEAL_FOUND)
    error("SEAL Not Found")
endif()