cmake_minimum_required(VERSION 3.6.3)

project(pybind11-download NONE)

include(ExternalProject)
if(IS_DIRECTORY "/mnt/DCVC/src/build/3rdparty/pybind11/pybind11-src/include")
    ExternalProject_Add(pybind11
        GIT_REPOSITORY    https://github.com/pybind/pybind11.git
        GIT_TAG           v2.6.1
        GIT_SHALLOW       1
        SOURCE_DIR        "/mnt/DCVC/src/build/3rdparty/pybind11/pybind11-src"
        BINARY_DIR        "/mnt/DCVC/src/build/3rdparty/pybind11/pybind11-build"
        DOWNLOAD_COMMAND  ""
        UPDATE_COMMAND    ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
    )
else()
    ExternalProject_Add(pybind11
        GIT_REPOSITORY    https://github.com/pybind/pybind11.git
        GIT_TAG           v2.6.1
        GIT_SHALLOW       1
        SOURCE_DIR        "/mnt/DCVC/src/build/3rdparty/pybind11/pybind11-src"
        BINARY_DIR        "/mnt/DCVC/src/build/3rdparty/pybind11/pybind11-build"
        UPDATE_COMMAND    ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
    )
endif()
