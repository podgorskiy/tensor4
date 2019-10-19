cmake_minimum_required(VERSION 3.6)

project(alexnet)

set(EXECUTABLE_NAME "alexnet")

if(MSVC)
    set(COMMON_OPTIONS /W3 /wd4996 /we4239 /we4002 /we4700 /we4305 /EHsc /MP)
    set(DEBUG_OPTIONS /MTd /Od /Zi)
    set(RELEASE_OPTIONS /MT /Ox /fp:fast /GL /GR-)
elseif(APPLE)
    set(COMMON_OPTIONS -DHAVE_PTHREAD)
    set(DEBUG_OPTIONS -g -Wall)
    set(RELEASE_OPTIONS -Ofast -fPIC -funsafe-math-optimizations -fno-strict-aliasing -fno-rtti -ffast-math -flto -msse2 -msse3 -msse4 -fopenmp)
else()
    set(COMMON_OPTIONS -DHAVE_PTHREAD)
    set(DEBUG_OPTIONS -g -Wall)
    set(RELEASE_OPTIONS -Ofast -fPIC -funsafe-math-optimizations -fno-strict-aliasing -fno-rtti -ffast-math -flto -msse2 -msse3 -msse4 -fopenmp)
endif()

set(DEBUG_OPTIONS ${DEBUG_OPTIONS} ${COMMON_OPTIONS})
set(RELEASE_OPTIONS ${RELEASE_OPTIONS} ${COMMON_OPTIONS})

add_compile_options(
        "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>"
        "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")

if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set (CMAKE_CXX_FLAGS "--std=c++14 ${CMAKE_CXX_FLAGS}")
    endif ()
else ()
    set (CMAKE_CXX_STANDARD 14)
endif ()

include_directories(../include)
include_directories(common)

# Main sources
##############################################################
file(GLOB SOURCES alexnet/*.cpp alexnet/*.h)
add_executable(alexnet ${SOURCES})

file(GLOB SOURCES densenet169/*.cpp densenet169/*.h)
add_executable(densenet169 ${SOURCES})

file(GLOB SOURCES vgg16/*.cpp vgg16/*.h)
add_executable(vgg16 ${SOURCES})

file(GLOB SOURCES resnet50/*.cpp resnet50/*.h)
add_executable(resnet50 ${SOURCES})

##############################################################
# Linkage
##############################################################
if(MSVC)
else()
    set(LIBRARIES rt m gomp)
endif()

target_link_libraries(alexnet ${LIBRARIES})
target_link_libraries(densenet169 ${LIBRARIES})
target_link_libraries(vgg16 ${LIBRARIES})
target_link_libraries(resnet50 ${LIBRARIES})
