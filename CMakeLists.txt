cmake_minimum_required(VERSION 3.24)

project(cuda_stream_fusion
    VERSION 0.1.0
    DESCRIPTION "Zero-copy CUDA multimedia stream fusion bindings for PyTorch"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build for")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CSF_ALLOW_UNSUPPORTED_HOST_COMPILER
    "Pass --allow-unsupported-compiler to nvcc for experimental host compiler/toolkit mixes"
    OFF
)

set(MMSF_PREFIX "" CACHE PATH "Path to the active micromamba/conda environment, normally named mmsf")

if(NOT MMSF_PREFIX)
    set(_mmsf_prefix_candidates)

    foreach(_env_var IN ITEMS CONDA_PREFIX MAMBA_PREFIX)
        if(DEFINED ENV{${_env_var}} AND EXISTS "$ENV{${_env_var}}/bin/python")
            list(APPEND _mmsf_prefix_candidates "$ENV{${_env_var}}")
        endif()
    endforeach()

    if(DEFINED ENV{MAMBA_ROOT_PREFIX})
        list(APPEND _mmsf_prefix_candidates "$ENV{MAMBA_ROOT_PREFIX}/envs/mmsf")
    endif()

    if(DEFINED ENV{HOME})
        list(APPEND _mmsf_prefix_candidates
            "$ENV{HOME}/micromamba/envs/mmsf"
            "$ENV{HOME}/mambaforge/envs/mmsf"
            "$ENV{HOME}/miniforge3/envs/mmsf"
            "$ENV{HOME}/miniconda3/envs/mmsf"
        )
    endif()

    foreach(_candidate IN LISTS _mmsf_prefix_candidates)
        if(EXISTS "${_candidate}/bin/python")
            set(MMSF_PREFIX "${_candidate}" CACHE PATH "Resolved mmsf environment prefix" FORCE)
            break()
        endif()
    endforeach()
endif()

if(NOT MMSF_PREFIX OR NOT EXISTS "${MMSF_PREFIX}/bin/python")
    message(FATAL_ERROR
        "Could not resolve the mmsf environment. Activate it before configuring, "
        "or pass -DMMSF_PREFIX=/absolute/path/to/env."
    )
endif()

message(STATUS "Using mmsf prefix: ${MMSF_PREFIX}")

list(PREPEND CMAKE_PREFIX_PATH
    "${MMSF_PREFIX}"
    "${MMSF_PREFIX}/lib/cmake"
    "${MMSF_PREFIX}/share/cmake"
)

set(Python3_ROOT_DIR "${MMSF_PREFIX}")
set(Python3_EXECUTABLE "${MMSF_PREFIX}/bin/python")
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    RESULT_VARIABLE _pybind11_query_rc
    OUTPUT_VARIABLE _pybind11_cmake_dir
    ERROR_VARIABLE _pybind11_query_err
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(_pybind11_query_rc EQUAL 0 AND EXISTS "${_pybind11_cmake_dir}")
    set(pybind11_DIR "${_pybind11_cmake_dir}" CACHE PATH "pybind11 CMake package directory" FORCE)
endif()

find_package(pybind11 CONFIG QUIET
    PATHS
        "${pybind11_DIR}"
        "${MMSF_PREFIX}"
        "${MMSF_PREFIX}/share/cmake/pybind11"
        "${MMSF_PREFIX}/lib/cmake/pybind11"
    NO_DEFAULT_PATH
)

find_package(CUDAToolkit REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
        "import site, sysconfig; paths = site.getsitepackages() + [sysconfig.get_path('purelib'), sysconfig.get_path('platlib')]; print(';'.join(p for p in paths if p))"
    RESULT_VARIABLE _site_query_rc
    OUTPUT_VARIABLE _python_site_paths
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(_pybind11_include_roots
    "${MMSF_PREFIX}/include"
    ${_python_site_paths}
)

find_path(PYBIND11_INCLUDE_DIR
    NAMES pybind11/pybind11.h
    PATHS ${_pybind11_include_roots}
    PATH_SUFFIXES pybind11/include include
    NO_DEFAULT_PATH
)

set(_fusion_core_sources
    src/main.cpp
    src/video_decoder.cu
)

if(pybind11_FOUND)
    pybind11_add_module(fusion_core MODULE ${_fusion_core_sources})
    target_link_libraries(fusion_core PRIVATE CUDA::cudart pybind11::module)
elseif(PYBIND11_INCLUDE_DIR)
    Python3_add_library(fusion_core MODULE WITH_SOABI ${_fusion_core_sources})
    target_include_directories(fusion_core PRIVATE "${PYBIND11_INCLUDE_DIR}")
    target_link_libraries(fusion_core PRIVATE CUDA::cudart Python3::Module)
else()
    message(FATAL_ERROR
        "pybind11 was not found inside the mmsf environment at ${MMSF_PREFIX}. "
        "Install or expose pybind11 in that environment, then re-run CMake. "
        "No system or global Python paths are used by this build."
    )
endif()

target_include_directories(fusion_core PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${MMSF_PREFIX}/include"
)

target_compile_definitions(fusion_core
    PRIVATE
        PYBIND11_DETAILED_ERROR_MESSAGES
)

target_compile_options(fusion_core
    PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall;-Wextra;-Wpedantic;-march=native>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--expt-relaxed-constexpr;--extended-lambda;-lineinfo>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${CSF_ALLOW_UNSUPPORTED_HOST_COMPILER}>>:--allow-unsupported-compiler>
)

set_target_properties(fusion_core PROPERTIES
    CUDA_ARCHITECTURES 89
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/fusion_engine"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/fusion_engine"
)

foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(fusion_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${_cfg} "${CMAKE_CURRENT_SOURCE_DIR}/fusion_engine"
        RUNTIME_OUTPUT_DIRECTORY_${_cfg} "${CMAKE_CURRENT_SOURCE_DIR}/fusion_engine"
    )
endforeach()

install(TARGETS fusion_core LIBRARY DESTINATION fusion_engine)
install(FILES fusion_engine/pipeline.py DESTINATION fusion_engine)
