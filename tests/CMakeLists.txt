find_package(Boost REQUIRED COMPONENTS unit_test_framework)

add_executable(TestJacobian TestJacobian.cpp)
target_link_libraries(TestJacobian mc_tasks_MinimumJerkTask Boost::unit_test_framework Boost::dynamic_linking)
add_test(TestJacobian TestJacobian)
