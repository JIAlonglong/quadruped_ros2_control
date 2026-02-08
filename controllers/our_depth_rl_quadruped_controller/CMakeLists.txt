cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
project(our_depth_rl_quadruped_controller)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)

# ---------------------- 手动定义 CUDA::nvToolsExt 目标 ----------------------
# 1. 指定 nvToolsExt 的头文件和库文件路径（根据你的实际路径修改）
set(NVTOOLSEXT_INCLUDE_DIR "/usr/local/cuda-12.9/targets/x86_64-linux/include/nvtx3")
set(NVTOOLSEXT_LIBRARY "/home/jialonglong/libtorch/lib/libnvToolsExt-847d78f2.so.1")

# 2. 验证路径有效性（确保文件存在）
if(NOT EXISTS "${NVTOOLSEXT_INCLUDE_DIR}/nvToolsExt.h")
  message(FATAL_ERROR "nvToolsExt 头文件不存在：${NVTOOLSEXT_INCLUDE_DIR}/nvToolsExt.h")
endif()
if(NOT EXISTS "${NVTOOLSEXT_LIBRARY}")
  message(FATAL_ERROR "nvToolsExt 库文件不存在：${NVTOOLSEXT_LIBRARY}")
endif()

# 3. 手动创建 CUDA::nvToolsExt 导入目标（关键步骤）
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
  set_target_properties(CUDA::nvToolsExt PROPERTIES
    IMPORTED_LOCATION "${NVTOOLSEXT_LIBRARY}"  # 库文件路径
    INTERFACE_INCLUDE_DIRECTORIES "${NVTOOLSEXT_INCLUDE_DIR}"  # 头文件路径
  )
endif()
# ----------------------------------------------------------------------

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)

find_package(yaml-cpp REQUIRED)

set(dependencies
        pluginlib
        rcpputils
        controller_interface
        realtime_tools
        control_input_msgs
        kdl_parser
        ament_index_cpp
        controller_common
        OpenCV
        cv_bridge
        tf2
        tf2_geometry_msgs
)

# find dependencies
foreach (Dependency IN ITEMS ${dependencies})
    find_package(${Dependency} REQUIRED)
endforeach ()

add_library(${PROJECT_NAME} SHARED
        src/RlQuadrupedController.cpp

        src/common/ObservationBuffer.cpp

        src/FSM/StateFixedStand.cpp
        src/FSM/StateRL.cpp

        src/control/LowPassFilter.cpp
        src/control/Estimator.cpp

        src/robot/QuadrupedRobot.cpp
        src/robot/RobotLeg.cpp

)
target_include_directories(${PROJECT_NAME}
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
        # "/usr/local/cuda-12.9/targets/x86_64-linux/include/nvtx3"
        PRIVATE
        ${YAML_CPP_INCLUDE_DIR}
        src)
target_link_libraries(${PROJECT_NAME} PUBLIC
        "${TORCH_LIBRARIES}"
        yaml-cpp
        # /home/jialonglong/libtorch/lib/libnvToolsExt-847d78f2.so.1
)
ament_target_dependencies(
        ${PROJECT_NAME} PUBLIC
        ${dependencies}
)

pluginlib_export_plugin_description_file(controller_interface our_depth_rl_quadruped_controller.xml)

install(
        DIRECTORY include/
        DESTINATION include
)

install(
        TARGETS ${PROJECT_NAME}
        EXPORT export_${PROJECT_NAME}
        ARCHIVE DESTINATION lib/${PROJECT_NAME}
        LIBRARY DESTINATION lib/${PROJECT_NAME}
        RUNTIME DESTINATION bin
)

install(
        DIRECTORY launch
        DESTINATION share/${PROJECT_NAME}/
)

install(
        DIRECTORY config
        DESTINATION share/${PROJECT_NAME}/
)

install(
        PROGRAMS
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_contact_gait.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_foot_force_threshold.py"
        DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(${dependencies})
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

if (BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    # the following line skips the linter which checks for copyrights
    # comment the line when a copyright and license is added to all source files
    set(ament_cmake_copyright_FOUND TRUE)
    # the following line skips cpplint (only works in a git repo)
    # comment the line when this package is in a git repo and when
    # a copyright and license is added to all source files
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif ()

ament_package()
