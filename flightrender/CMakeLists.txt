cmake_minimum_required(VERSION 3.8)
project(flightrender)

find_package(ament_cmake REQUIRED)

# Install the Unity executable and its data
install(
  PROGRAMS RPG_Flightmare.x86_64
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY RPG_Flightmare_Data/
  DESTINATION lib/${PROJECT_NAME}/RPG_Flightmare_Data
)

install(
  FILES UnityPlayer.so
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()