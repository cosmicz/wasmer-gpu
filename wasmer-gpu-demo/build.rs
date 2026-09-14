fn main() {
    let host = std::env::var("HOST").expect("Cargo sets HOST for build scripts");
    println!("cargo:rustc-env=BUILD_HOST_TRIPLE={host}");
}
