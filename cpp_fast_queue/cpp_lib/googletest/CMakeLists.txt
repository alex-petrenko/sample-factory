project(googletest)

add_subdirectory(googletest-1.10.0)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(tests_run tests.cpp)
target_link_libraries(tests_run gtest gtest_main)
target_link_libraries(tests_run fast_queue)
