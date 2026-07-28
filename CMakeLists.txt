cmake_minimum_required(VERSION 3.12)
project(SupeRANSAC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Optional fast-math (OFF by default) ----
option(SUPERANSAC_ENABLE_FAST_MATH "Enable aggressive fast-math" ON)

# ---- Flag probes ----
# 包含 CheckCXXCompilerFlag 模块以检查编译器是否支持特定的编译标志
include(CheckCXXCompilerFlag)
function(add_flag_if_supported flag)
  # flag_var -存储结果的变量名，flag - 要处理的原始字符串
  string(REPLACE "=" "_" flag_var "${flag}")
  string(REGEX REPLACE "^-+" "" flag_var "${flag_var}")
  string(REPLACE "-" "_" flag_var "${flag_var}")
  # 检查编译器是否支持该标志，并将结果存储在 HAS_${flag_var} 中
  check_cxx_compiler_flag("${flag}" "HAS_${flag_var}")
  if(HAS_${flag_var})
    add_compile_options($<$<CONFIG:Release>:${flag}>)
  endif()
endfunction()

# ---- Baseline high-performance flags ----
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_flag_if_supported(-O3)
  add_flag_if_supported(-march=native)
  add_flag_if_supported(-mtune=native)  # 不启用专有指令集，兼容其他 CPU
  # PLT -过程链接表，作用是延迟绑定，首次调用时，动态链接器解析实际函数地址，后续调用直接跳转到已解析的地址
  # PLT 启动时间较快，首次调用有额外开销，内存使用较少，适合大多数情况
  add_flag_if_supported(-fno-plt)
  add_flag_if_supported(-funroll-loops)  # 循环展开，小循环明显加速，但代码体积变大
  add_flag_if_supported(-ftree-vectorize)  # 自动 SIMD
  add_flag_if_supported(-fomit-frame-pointer)
  add_flag_if_supported(-finline-functions)  # 更加积极的内联函数
  add_flag_if_supported(-fprefetch-loop-arrays)  #  预取循环数组
  add_flag_if_supported(-fno-semantic-interposition)  #
  add_flag_if_supported(-fno-trapping-math)  # 允许编译器进行更激进的数学优化
  add_compile_options($<$<CONFIG:Release>:-DNDEBUG>)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_flag_if_supported(-O3)
  add_flag_if_supported(-march=native)
  add_flag_if_supported(-mtune=native)
  check_cxx_compiler_flag("-flto=thin" HAS_FLTO_THIN)
  if(HAS_FLTO_THIN)
    add_compile_options($<$<CONFIG:Release>:-flto=thin>)
    add_link_options($<$<CONFIG:Release>:-flto=thin>)
  else()
    add_flag_if_supported(-flto)
    add_link_options($<$<CONFIG:Release>:-flto>)
  endif()
  add_flag_if_supported(-funroll-loops)
  add_flag_if_supported(-fomit-frame-pointer)
  add_flag_if_supported(-finline-functions)
  add_compile_options($<$<CONFIG:Release>:-DNDEBUG>)
elseif(MSVC)
  add_compile_options($<$<CONFIG:Release>:/O2 /Oi /GL /Gy /DNDEBUG /MP /Zc:inline>)
  add_link_options($<$<CONFIG:Release>:/LTCG>)
endif()

# ---- Optional fast-math flags ----
if(SUPERANSAC_ENABLE_FAST_MATH)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_flag_if_supported(-ffast-math)  # 综合数学优化
    add_flag_if_supported(-fno-math-errno)  # 禁用错误检测
    add_flag_if_supported(-ffp-contract=fast)  # 允许编译器将多个浮点操作合并为单个操作
  elseif(MSVC)
    add_compile_options($<$<CONFIG:Release>:/fp:fast>)
  endif()
endif()

# ---- IPO/LTO in a portable way ----，检查是否支持IPO（Interprocedural Optimization）或LTO（Link Time Optimization）
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_msg)
if(ipo_supported)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

# ---- Eigen: disable debug in release ----
add_compile_definitions($<$<CONFIG:Release>:EIGEN_NO_DEBUG>)

# ---- Modern pybind11 finder hint (silences CMP0148 warning downstream) ----
set(PYBIND11_FINDPYTHON ON)

# ---- Optional policy for Boost finder (silences CMP0167) ----
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

add_subdirectory(src)
add_subdirectory(python)
