# SPDK requires DPDK which must be built with meson, then SPDK with configure/make
# System dependencies required: uuid-dev (libuuid1), libnuma-dev

# Check for required system libraries
#find_package(PkgConfig REQUIRED)
#pkg_check_modules(UUID REQUIRED uuid)
#pkg_check_modules(NUMA REQUIRED numa)

fetchcontent_declare(
        spdk
        GIT_REPOSITORY https://github.com/spdk/spdk
        GIT_TAG v24.09
)

fetchcontent_getproperties(spdk)
if(NOT spdk_POPULATED)
    fetchcontent_populate(spdk)

    # Get processor count
    include(ProcessorCount)
    ProcessorCount(NPROC)
    if(NPROC EQUAL 0)
        set(NPROC 1)
    endif()

    # Initialize submodules
    add_custom_command(
            OUTPUT "${spdk_SOURCE_DIR}/.submodules_done"
            COMMAND git submodule update --init --recursive
            COMMAND touch .submodules_done
            WORKING_DIRECTORY ${spdk_SOURCE_DIR}
            COMMENT "Initializing SPDK submodules"
            VERBATIM
    )

    # Build DPDK with meson - minimal build, no network drivers needed for SPDK NVMe
    add_custom_command(
            OUTPUT "${spdk_SOURCE_DIR}/dpdk/install/lib64/librte_eal.a"
            COMMAND meson setup --prefix=${spdk_SOURCE_DIR}/dpdk/install
                    -Dplatform=native
                    -Denable_kmods=false
                    -Dtests=false
                    -Ddisable_drivers=net/*,crypto/*,compress/*,regex/*,vdpa/*,event/*,baseband/*,raw/*
                    -Denable_drivers=bus/pci,bus/vdev,mempool/ring
                    build
            COMMAND ninja -C build -j${NPROC}
            COMMAND meson install -C build
            WORKING_DIRECTORY ${spdk_SOURCE_DIR}/dpdk
            DEPENDS "${spdk_SOURCE_DIR}/.submodules_done"
            COMMENT "Building DPDK (minimal for SPDK NVMe)"
            VERBATIM
    )

    # Configure SPDK - minimal build for NVMe PCIe only
    add_custom_command(
            OUTPUT "${spdk_SOURCE_DIR}/mk/config.mk"
            COMMAND ${CMAKE_COMMAND} -E env
                    PKG_CONFIG_PATH=${spdk_SOURCE_DIR}/dpdk/install/lib64/pkgconfig:$ENV{PKG_CONFIG_PATH}
                    ./configure
                    --without-vhost
                    --without-crypto
                    --without-rbd
                    --without-fc
                    --without-iscsi-initiator
                    --without-vtune
                    --without-ocf
                    --without-fuse
                    --without-nvme-cuse
                    --with-shared
                    --with-dpdk=${spdk_SOURCE_DIR}/dpdk/install
                    --disable-tests
                    --disable-unit-tests
                    --disable-examples
                    --disable-apps
            WORKING_DIRECTORY ${spdk_SOURCE_DIR}
            DEPENDS "${spdk_SOURCE_DIR}/dpdk/install/lib64/librte_eal.a"
            COMMENT "Configuring SPDK (no tests/examples)"
    )

    # Build SPDK
    add_custom_command(
            OUTPUT "${spdk_SOURCE_DIR}/build/lib/libspdk_nvme.a"
            COMMAND make -j${NPROC}
            WORKING_DIRECTORY ${spdk_SOURCE_DIR}
            DEPENDS "${spdk_SOURCE_DIR}/mk/config.mk"
            COMMENT "Building SPDK"
            VERBATIM
    )

    add_custom_target(build_spdk_target
            DEPENDS "${spdk_SOURCE_DIR}/build/lib/libspdk_nvme.a"
    )

    # Create imported libraries
    add_library(spdk_nvme STATIC IMPORTED GLOBAL)
    set_target_properties(spdk_nvme PROPERTIES
            IMPORTED_LOCATION "${spdk_SOURCE_DIR}/build/lib/libspdk_nvme.a"
    )
    add_dependencies(spdk_nvme build_spdk_target)

    add_library(spdk_env_dpdk STATIC IMPORTED GLOBAL)
    set_target_properties(spdk_env_dpdk PROPERTIES
            IMPORTED_LOCATION "${spdk_SOURCE_DIR}/build/lib/libspdk_env_dpdk.a"
    )
    add_dependencies(spdk_env_dpdk build_spdk_target)

    # Combined interface - link DPDK static libraries directly
    set(DPDK_LIB_DIR "${spdk_SOURCE_DIR}/dpdk/install/lib64")

    add_library(spdk_interface INTERFACE)
    target_link_libraries(spdk_interface INTERFACE
            -Wl,--whole-archive
            spdk_nvme
            spdk_env_dpdk
            "${spdk_SOURCE_DIR}/build/lib/libspdk_util.a"
            "${spdk_SOURCE_DIR}/build/lib/libspdk_log.a"
            "${spdk_SOURCE_DIR}/build/lib/libspdk_sock.a"
            "${spdk_SOURCE_DIR}/build/lib/libspdk_sock_posix.a"
            "${spdk_SOURCE_DIR}/build/lib/libspdk_trace.a"
            "${spdk_SOURCE_DIR}/build/lib/libspdk_dma.a"
            "${spdk_SOURCE_DIR}/build/lib/libspdk_keyring.a"
            "${spdk_SOURCE_DIR}/build/lib/libspdk_json.a"
            "${spdk_SOURCE_DIR}/build/lib/libspdk_jsonrpc.a"
            "${spdk_SOURCE_DIR}/build/lib/libspdk_rpc.a"
            "${spdk_SOURCE_DIR}/build/lib/libspdk_thread.a"
            "${DPDK_LIB_DIR}/librte_log.a"
            "${DPDK_LIB_DIR}/librte_eal.a"
            "${DPDK_LIB_DIR}/librte_mempool.a"
            "${DPDK_LIB_DIR}/librte_mempool_ring.a"
            "${DPDK_LIB_DIR}/librte_ring.a"
            "${DPDK_LIB_DIR}/librte_mbuf.a"
            "${DPDK_LIB_DIR}/librte_bus_pci.a"
            "${DPDK_LIB_DIR}/librte_pci.a"
            "${DPDK_LIB_DIR}/librte_kvargs.a"
            "${DPDK_LIB_DIR}/librte_telemetry.a"
            -Wl,--no-whole-archive
            pthread rt uuid numa dl crypto ssl
    )
    target_include_directories(spdk_interface INTERFACE
            "${spdk_SOURCE_DIR}/include"
            "${spdk_SOURCE_DIR}/dpdk/install/include"
    )
    add_dependencies(spdk_interface build_spdk_target)

    message(STATUS "SPDK will be built from source")
endif()
