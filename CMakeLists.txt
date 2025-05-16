cmake_minimum_required(VERSION 3.10)

project(MyGoCppLibTorchApp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB PYTHON_TORCH_PATHS "${CMAKE_CURRENT_LIST_DIR}/python/venv/lib/python3*/site-packages/torch/")
list(JOIN PYTHON_TORCH_PATHS ";" CMAKE_PREFIX_PATH)
set(CMAKE_PREFIX_PATH "/usr/local/libtorch;${PYTHON_TORCH_PATHS}")

find_package(Torch REQUIRED)

find_package(CUDA QUIET)

if(CUDA_FOUND)
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    set(CUDA_LIBRARIES cuda cudart ${CUDA_LIBRARIES})
else()
    message(STATUS "CUDA not found, skipping CUDA")
    set(CUDA_LIBRARIES "")
endif()

link_directories(${LIBTORCH_PATH}/lib)
add_library(torch_bind SHARED ../c++/torch_bind.cpp ../c++/torch.cpp)

set_target_properties(torch_bind PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)

target_link_libraries(torch_bind
    torch
    c10
    ${CUDA_LIBRARIES}
    pthread
)

# Models
add_library(mnist SHARED ../c++/models/mnist.cpp)
set_target_properties(mnist PROPERTIES OUTPUT_NAME "mnist")
set_target_properties(mnist PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/models)
set_target_properties(mnist PROPERTIES SUFFIX ".so")

target_link_libraries(mnist
    torch_bind
    torch
    c10
    ${CUDA_LIBRARIES}
    pthread
)

# Determine platform-specific shared lib name and rpath flag
if(APPLE)
    set(RPATH_FLAG "-Wl,-rpath,${CMAKE_CURRENT_SOURCE_DIR}/build")
elseif(UNIX)
    set(RPATH_FLAG "-Wl,-rpath=${CMAKE_CURRENT_SOURCE_DIR}/build")
endif()

# Go app
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/build/goTorch
    COMMAND ${CMAKE_COMMAND} -E env
       CGO_CPPFLAGS=-I${CMAKE_CURRENT_SOURCE_DIR}/c++
       CGO_LDFLAGS=-L${CMAKE_CURRENT_SOURCE_DIR}/build\ -ltorch_bind\ ${RPATH_FLAG}
       -- go build -o ${CMAKE_CURRENT_SOURCE_DIR}/build/goTorch
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/go
    DEPENDS torch_bind mnist
    COMMENT "Building Go app with libtorch C++ bindings"
)
       # -- python3 -c 'import os\; print(os.environ.get(\"CGO_CXXFLAGS\")+\";\"+os.environ.get(\"CGO_LDFLAGS\"))'

add_custom_target(buildGo ALL
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/build/goTorch
)


#set(GO_SOURCES main.go)
#set(GO_LDFLAGS -L${LIBRARY_OUTPUT_DIRECTORY})

#add_custom_target(
    #main ALL
    #COMMAND go build -o main ${GO_SOURCES}
    #WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#)
#
#add_dependencies(main test_cnn)
#add_dependencies(torch_bind)
