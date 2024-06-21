# mc_tasks

set(mc_tasks_SRC
  mc_tasks/MinimumJerkTask.cpp
)

set(mc_tasks_HDR
  ../include/mc_tasks/MinimumJerkTask.h
)

add_library(mc_tasks_${PROJECT_NAME} SHARED ${mc_tasks_SRC} ${mc_tasks_HDR})

target_link_libraries(mc_tasks_${PROJECT_NAME} PUBLIC mc_rtc::mc_tasks)

set_target_properties(mc_tasks_${PROJECT_NAME} PROPERTIES
  OUTPUT_NAME mc_tasks_${PROJECT_NAME}
  VERSION mc_tasks_${PROJECT_NAME}
  PUBLIC_HEADER "${mc_tasks_${PROJECT_NAME}_HDR}")

target_include_directories(mc_tasks_${PROJECT_NAME}
  PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/mc_task>"
  "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>"
)

install(TARGETS mc_tasks_${PROJECT_NAME}
  EXPORT "${TARGETS_EXPORT_NAME}"
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
