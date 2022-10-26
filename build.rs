// BSL 1.0/Apache 2.0 License

//! The build flags generated by this script are not public API.

use std::env;
use std::process::Command;

fn main() {
    // Run rustc to get the version.
    let rustc_path = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let rustc_command = Command::new(rustc_path)
        .arg("--version")
        .arg("--verbose")
        .output();

    match rustc_command {
        Ok(output) => {
            if !output.status.success() {
                println!(
                    "cargo:warning=async-io: rustc failed with error code {}",
                    output.status.code().unwrap_or(0)
                );
                return;
            }

            let version = String::from_utf8_lossy(&output.stdout);

            // Quick and dirty: Don't set the flag if we don't detect the string "nightly".
            if !version.contains("nightly") {
                // We set the inverse flag here for compatibility with non-Cargo build
                // systems. In this case, if the flag is not set, we assume that we are
                // running on a non-nightly system and just fall back to basic primitives.
                println!("cargo:rustc-cfg=breadsimd_no_nightly");
            }
        }
        Err(err) => {
            // Notify the user of the warning.
            println!(
                "cargo:warning=breadsimd: failed to detect compiler version: {}",
                err
            );
        }
    }
}
