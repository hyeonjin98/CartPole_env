cmake_minimum_required(VERSION 3.10)

# 프로젝트 이름과 버전 설정
project(MyProject VERSION 1.0)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Python 경로 설정
set(Python3_ROOT_DIR /home/hyeonjin/anaconda3/envs/sim39)
set(Python3_EXECUTABLE /home/hyeonjin/anaconda3/envs/sim39/bin/python)
find_package(PythonInterp 3.9 REQUIRED)
find_package(PythonLibs 3.9 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

# pybind11 서브 디렉토리 추가
add_subdirectory(pybind11)

# Eigen 라이브러리 찾기 및 경로 포함
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# yaml-cpp 라이브러리 찾기 및 경로 포함
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

# pybind11 모듈 생성
pybind11_add_module(cart_pole_module cart_pole_bindings.cpp cart_pole.cpp)

# pybind11 모듈에 yaml-cpp 및 Eigen 라이브러리 링크
target_link_libraries(cart_pole_module PRIVATE ${YAML_CPP_LIBRARIES} ${PYTHON_LIBRARIES} Eigen3::Eigen)

# cfg.yaml 파일을 빌드 디렉토리로 복사하는 규칙 추가
configure_file(${CMAKE_SOURCE_DIR}/cfg.yaml ${CMAKE_BINARY_DIR}/cfg.yaml COPYONLY)

# 빌드 후 LD_LIBRARY_PATH 설정 추가 (필요 시)
set_target_properties(cart_pole_module PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
