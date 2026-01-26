use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "vendored")]
    let include_path = build_vendored();

    #[cfg(not(feature = "vendored"))]
    let include_path = find_system_library();

    generate_bindings(&include_path);
}

#[cfg(feature = "vendored")]
fn build_vendored() -> PathBuf {
    // Get the project root (3 levels up from hmll-sys: lib/rust/hmll-sys -> .)
    let project_root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    println!("cargo:rerun-if-changed=../../..");

    // Detect Rust build profile and map to CMake build type
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let cmake_build_type = match profile.as_str() {
        "debug" => "Debug",
        "release" => "Release",
        "bench" => "Release",
        _ => "RelWithDebInfo",
    };

    // Configure CMake build
    let mut cmake_config = cmake::Config::new(&project_root);

    // Set the CMake build type based on Rust profile
    cmake_config.profile(cmake_build_type);

    // Enable features based on Rust features
    cmake_config
        .define("HMLL_BUILD_STATIC", "ON")
        .define("HMLL_BUILD_EXAMPLES", "OFF")
        .define("HMLL_BUILD_TESTS", "OFF")
        .define("HMLL_ENABLE_PYTHON", "OFF")
        .build_target("libhmll");

    // io_uring is Linux-only
    #[cfg(all(target_os = "linux", feature = "io_uring"))]
    cmake_config.define("HMLL_ENABLE_IO_URING", "ON");

    #[cfg(not(all(target_os = "linux", feature = "io_uring")))]
    cmake_config.define("HMLL_ENABLE_IO_URING", "OFF");

    #[cfg(feature = "safetensors")]
    cmake_config.define("HMLL_ENABLE_SAFETENSORS", "ON");

    #[cfg(not(feature = "safetensors"))]
    cmake_config.define("HMLL_ENABLE_SAFETENSORS", "OFF");

    #[cfg(feature = "cuda")]
    cmake_config.define("HMLL_ENABLE_CUDA", "ON");

    #[cfg(not(feature = "cuda"))]
    cmake_config.define("HMLL_ENABLE_CUDA", "OFF");

    // Build the library
    let dst = cmake_config.build();

    // Tell cargo to link the library
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=static=libhmll");

    // Link io_uring if enabled
    #[cfg(all(target_os = "linux", feature = "io_uring"))]
    {
        println!(
            "cargo:rustc-link-search=native={}/build/_deps/liburing-src/src",
            dst.display()
        );
        println!("cargo:rustc-link-lib=static=uring");
    }

    // Link yyjson if safetensors is enabled
    #[cfg(feature = "safetensors")]
    {
        println!(
            "cargo:rustc-link-search=native={}/build/_deps/yyjson-build",
            dst.display()
        );
        println!("cargo:rustc-link-lib=static=yyjson");
    }

    // Link CUDA runtime if enabled
    #[cfg(feature = "cuda")]
    link_cuda();

    project_root.join("include")
}

#[cfg(not(feature = "vendored"))]
fn find_system_library() -> PathBuf {
    // Try pkg-config first
    let library = pkg_config::Config::new()
        .atleast_version("0.1.0")
        .probe("hmll")
        .expect(
            "Could not find system hmll library. \
             Either install libhmll or enable the 'vendored' feature to build from source.",
        );

    // pkg-config handles linking automatically, but we need additional libraries
    // based on features

    #[cfg(all(target_os = "linux", feature = "io_uring"))]
    {
        // Try to find liburing via pkg-config, fall back to direct linking
        if pkg_config::probe_library("liburing").is_err() {
            println!("cargo:rustc-link-lib=uring");
        }
    }

    #[cfg(feature = "safetensors")]
    {
        // yyjson is typically statically linked into hmll, but try pkg-config
        let _ = pkg_config::probe_library("yyjson");
    }

    #[cfg(feature = "cuda")]
    link_cuda();

    // Return the first include path from pkg-config, or fall back to standard paths
    library
        .include_paths
        .first()
        .cloned()
        .unwrap_or_else(|| PathBuf::from("/usr/include"))
}

#[cfg(feature = "cuda")]
fn link_cuda() {
    // Try to find CUDA installation
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-search=native={}/lib", cuda_path);
    } else if let Ok(cuda_home) = env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
        println!("cargo:rustc-link-search=native={}/lib", cuda_home);
    } else {
        // Try common default locations
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib");
        println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
        println!("cargo:rustc-link-search=native=/opt/cuda/lib");
    }

    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
}

fn generate_bindings(include_path: &Path) {
    let builder = bindgen::Builder::default()
        .header(include_path.join("hmll/hmll.h").to_str().unwrap())
        .clang_arg(format!("-I{}", include_path.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("hmll_.*")
        .allowlist_type("hmll_.*")
        .allowlist_var("HMLL_.*")
        .derive_debug(true)
        .derive_default(true)
        .derive_copy(true)
        .derive_eq(true)
        .derive_hash(true)
        .no_partialeq("hmll_loader")
        // Exclude C stdlib types with function pointers from PartialEq/Eq
        // (comparing function pointers is undefined behavior)
        .no_partialeq("__sFILE")
        .impl_debug(true)
        .prepend_enum_name(false)
        .size_t_is_usize(true)
        .layout_tests(false)
        .rustified_enum("hmll_status_code")
        .rustified_enum("hmll_device")
        .rustified_enum("hmll_loader_kind")
        .rustified_enum("hmll_dtype");

    // Add conditional defines based on features
    #[cfg(feature = "safetensors")]
    let builder = builder
        .clang_arg("-D__HMLL_SAFETENSORS_ENABLED__=1")
        .clang_arg("-D__HMLL_TENSORS_ENABLED__=1");

    #[cfg(feature = "cuda")]
    let builder = builder.clang_arg("-D__HMLL_CUDA_ENABLED__=1");

    let bindings = builder.generate().expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
