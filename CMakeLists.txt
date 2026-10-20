#################################################################################
#                                                                               #
#  Copyright (C) 2026 Seif Kandil (k0T0z)                                       #
#                                                                               #
#  This file is a part of the ENIGMA Development Environment.                   #
#                                                                               #
#                                                                               #
#  ENIGMA is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software    #
#  Foundation, version 3 of the license or any later version.                   #
#                                                                               #
#  This application and its source code is distributed AS-IS, WITHOUT ANY       #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS    #
#  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more        #
#  details.                                                                     #
#                                                                               #
#  You should have recieved a copy of the GNU General Public License along      #
#  with this code. If not, see <http://www.gnu.org/licenses/>                   #
#                                                                               #
#  ENIGMA is an environment designed to create games and other programs with a  #
#  high-level, fully compilable language. Developers of ENIGMA or anything      #
#  associated with ENIGMA are in no way responsible for its users or            #
#  applications created by its users, or damages caused by the environment      #
#  or programs made in the environment.                                         #
#                                                                               #
#################################################################################

####################################################################################
#  SGP - the Shader Generation Problem.
#
#  Python bindings for shader-gen's `generator` module. shader-gen is consumed as a git
#  submodule and only its Qt-free generator sources are compiled here: SGP deliberately
#  does not use shader-gen's CMakeLists, so no Qt and no GUI code is pulled in.
#
#  Configure (from the SGP root):
#
#    cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
#          -DCMAKE_PREFIX_PATH="<protobuf-prefix>;<absl-prefix>" ^
#          -Dpybind11_DIR="<pybind11-cmake-dir>"
#
#  where <protobuf-prefix> holds lib/cmake/protobuf and lib/cmake/utf8_range, while
#  <absl-prefix> holds lib/cmake/absl. `python -m pybind11 --cmakedir` prints the latter.
####################################################################################
cmake_minimum_required(VERSION 3.21)

project(SGP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#####################
# shader-gen sources
#####################

set(SGP_SHADER_GEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/externals/shader-gen"
    CACHE PATH "shader-gen checkout that provides the generator sources.")

set(SGP_SCHEMA_DIR "${SGP_SHADER_GEN_ROOT}/gui/model/schema")

if(NOT EXISTS "${SGP_SCHEMA_DIR}/visual_shader.proto")
    message(FATAL_ERROR
        "shader-gen sources not found under '${SGP_SHADER_GEN_ROOT}'. Fetch the submodule with:\n"
        "  git submodule update --init --recursive")
endif()

find_package(Protobuf CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

#####################
# Protobuf schema
#####################

set(SGP_PROTO_FILES
    ${SGP_SCHEMA_DIR}/visual_shader.proto
    ${SGP_SCHEMA_DIR}/visual_shader_nodes.proto
)

# `protoc` writes the generated files relative to their import root while shader-gen's
# generator headers include them as "gui/model/schema/<name>.pb.h". Mirror that layout so
# the same include paths resolve. Generating into a single static library (rather than per
# target) keeps protoc from racing on the same outputs during parallel builds.
set(SGP_PROTOC_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/gui/model/schema")
file(MAKE_DIRECTORY "${SGP_PROTOC_OUT_DIR}")

add_library(sgp_proto STATIC ${SGP_PROTO_FILES})

protobuf_generate(
    TARGET sgp_proto
    LANGUAGE cpp
    PROTOC_OUT_DIR "${SGP_PROTOC_OUT_DIR}"
    IMPORT_DIRS "${SGP_SCHEMA_DIR}" # For the imports in the proto files
)

target_link_libraries(sgp_proto PUBLIC protobuf::libprotobuf)

target_include_directories(sgp_proto PUBLIC
    "${CMAKE_CURRENT_BINARY_DIR}" # For the generated protobuf files
    "${SGP_SHADER_GEN_ROOT}"      # For "error_macros.hpp"
)

#####################
# Generator
#####################

set(SGP_GENERATOR_SOURCES
    ${SGP_SHADER_GEN_ROOT}/generator/visual_shader_generator.cpp
    ${SGP_SHADER_GEN_ROOT}/generator/visual_shader_node_generators.cpp
    ${SGP_SHADER_GEN_ROOT}/generator/vs_node_noise_generators.cpp
    ${SGP_SHADER_GEN_ROOT}/generator/visual_shader_node_port_type_generator.cpp
)

add_library(sgp_generator STATIC ${SGP_GENERATOR_SOURCES})

target_link_libraries(sgp_generator PUBLIC sgp_proto)

target_include_directories(sgp_generator PUBLIC
    "${SGP_SHADER_GEN_ROOT}" # For "generator/..."
)

#####################
# Python module
#####################

pybind11_add_module(shader_gen_py src/bindings.cpp)

target_link_libraries(shader_gen_py PRIVATE sgp_generator)

# Collect the module in a stable location so `main.py` can import it without knowing the
# generator-specific subdirectory CMake would otherwise pick.
set(SGP_PYTHON_DIR "${CMAKE_BINARY_DIR}/python")
foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(shader_gen_py PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY_${_config} "${SGP_PYTHON_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_${_config} "${SGP_PYTHON_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_${_config} "${SGP_PYTHON_DIR}")
endforeach()
set_target_properties(shader_gen_py PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${SGP_PYTHON_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${SGP_PYTHON_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${SGP_PYTHON_DIR}")

# Protobuf is built as a shared library; ship it and its own runtime dependencies next to
# the module so `import shader_gen_py` resolves them without touching PATH.
if(WIN32)
    set(_sgp_runtime_dlls "$<TARGET_FILE:protobuf::libprotobuf>")
    foreach(_dep utf8_range::utf8_validity absl::abseil_dll)
        if(TARGET ${_dep})
            list(APPEND _sgp_runtime_dlls "$<TARGET_FILE:${_dep}>")
        endif()
    endforeach()

    add_custom_command(TARGET shader_gen_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_sgp_runtime_dlls} "$<TARGET_FILE_DIR:shader_gen_py>"
        VERBATIM)
endif()
