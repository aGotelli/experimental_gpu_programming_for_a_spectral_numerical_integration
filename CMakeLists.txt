cmake_minimum_required(VERSION 3.16)

project(spectral_integration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)


include_directories(include)

add_executable(main
    main.cpp
    include/chebyshev_differentiation.h
    include/utilities.h
)
target_link_libraries(main
    PUBLIC
        Eigen3::Eigen
)




#add_executable(spectral_integration main.cpp
#                                    include/tictoc.h
#                                    include/chebyshev_differentiation.h
#                                    include/spectral_integration_utilities.h)
#target_link_libraries(spectral_integration Eigen3::Eigen)





#add_executable(revised_integration_quaternion src/revised_integration_quaternion.cpp)
#target_link_libraries(revised_integration_quaternion Eigen3::Eigen benchmark::benchmark)

#add_executable(revised_integration_position src/revised_integration_position.cpp)
#target_link_libraries(revised_integration_position Eigen3::Eigen benchmark::benchmark)
