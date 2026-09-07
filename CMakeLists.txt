cmake_minimum_required(VERSION 3.20)

project(pe115_pillarcracker LANGUAGES CXX)

add_executable(pe115_pillarcracker
    src/main.cpp
)

target_compile_features(pe115_pillarcracker PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(pe115_pillarcracker PRIVATE /W4 /permissive-)
else()
    target_compile_options(pe115_pillarcracker PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Keep AVX2 code in a separate translation unit.  The executable dispatches to
# it only after a runtime CPU/OS capability check, leaving the scalar path
# usable on older machines.
include(CheckCXXCompilerFlag)
if(MSVC)
    check_cxx_compiler_flag("/arch:AVX2" PE115_HAS_AVX2_FLAG)
    set(PE115_AVX2_FLAG "/arch:AVX2")
else()
    check_cxx_compiler_flag("-mavx2" PE115_HAS_AVX2_FLAG)
    set(PE115_AVX2_FLAG "-mavx2")
endif()

if(PE115_HAS_AVX2_FLAG)
    add_library(pe115_pillarcracker_avx2 OBJECT src/avx2.cpp)
    target_compile_features(pe115_pillarcracker_avx2 PRIVATE cxx_std_20)
    target_compile_options(pe115_pillarcracker_avx2 PRIVATE ${PE115_AVX2_FLAG})
    if(MSVC)
        target_compile_options(pe115_pillarcracker_avx2 PRIVATE /W4 /permissive-)
    else()
        target_compile_options(pe115_pillarcracker_avx2 PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    target_sources(pe115_pillarcracker PRIVATE
        $<TARGET_OBJECTS:pe115_pillarcracker_avx2>)
    target_compile_definitions(pe115_pillarcracker PRIVATE PE115_HAVE_AVX2=1)
endif()

# CUDA is also optional.  Its kernel evaluates the same 32-bit candidate
# predicate as the CPU backends, which is useful for an intentionally explicit
# whole-seed-space search on NVIDIA hardware.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # CMake learned the portable "native" value in 3.24.  On older CMake
    # releases retain a conservative architecture that can still run on newer
    # NVIDIA GPUs; callers may set PE115_CUDA_ARCHITECTURES explicitly.
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(PE115_DEFAULT_CUDA_ARCHITECTURES native)
    else()
        set(PE115_DEFAULT_CUDA_ARCHITECTURES 75)
    endif()
    set(PE115_CUDA_ARCHITECTURES "${PE115_DEFAULT_CUDA_ARCHITECTURES}"
        CACHE STRING "CUDA architectures for the optional CUDA backend")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "${PE115_CUDA_ARCHITECTURES}")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(pe115_pillarcracker_cuda OBJECT src/cuda_backend.cu)
    target_compile_features(pe115_pillarcracker_cuda PRIVATE cxx_std_20)
    set_target_properties(pe115_pillarcracker_cuda PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_sources(pe115_pillarcracker PRIVATE
        $<TARGET_OBJECTS:pe115_pillarcracker_cuda>)
    target_link_libraries(pe115_pillarcracker PRIVATE CUDA::cudart)
    target_compile_definitions(pe115_pillarcracker PRIVATE PE115_HAVE_CUDA=1)
    set_target_properties(pe115_pillarcracker PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()

# The scanner remains correct without OpenMP.  When it is available, split an
# explicitly selected seed range across CPU cores.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pe115_pillarcracker PRIVATE OpenMP::OpenMP_CXX)
    if(TARGET pe115_pillarcracker_avx2)
        target_link_libraries(pe115_pillarcracker_avx2
            PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
