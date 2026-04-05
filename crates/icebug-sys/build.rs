use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let icebug_source = manifest_dir
        .join("..")
        .join("..")
        .join("vendor")
        .join("icebug")
        .canonicalize()
        .expect("vendor/icebug not found — did you init git submodules?");

    let dst = cmake::Config::new(manifest_dir.join("wrapper"))
        .define("ICEBUG_SOURCE_DIR", &icebug_source)
        .build_target("icebug_c")
        .build();

    let build_dir = dst.join("build");

    // Link our wrapper
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=icebug_c");

    // Link networkit built by cmake
    // It may be in build/icebug/ or build/icebug/networkit/
    for subdir in &["icebug", "icebug/networkit", "."] {
        let p = build_dir.join(subdir);
        if p.exists() {
            println!("cargo:rustc-link-search=native={}", p.display());
        }
    }
    println!("cargo:rustc-link-lib=static=networkit");

    // Link C++ standard library
    let target = env::var("TARGET").unwrap();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // Link OpenMP if available
    if target.contains("apple") {
        // On macOS with Homebrew's libomp
        if let Ok(output) = std::process::Command::new("brew")
            .args(["--prefix", "libomp"])
            .output()
        {
            if output.status.success() {
                let prefix = String::from_utf8_lossy(&output.stdout).trim().to_string();
                println!("cargo:rustc-link-search=native={}/lib", prefix);
                println!("cargo:rustc-link-lib=dylib=omp");
            }
        }
    } else {
        println!("cargo:rustc-link-lib=dylib=gomp");
    }
}
