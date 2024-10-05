use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let proto_file = "./proto/stt/transcribe.proto";
    let proto_gen_dir = "./src/pb";

    std::fs::create_dir_all(proto_gen_dir)?;

    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .build_server(true)
        .build_client(false)
        .file_descriptor_set_path(out_dir.join("transcribe_descriptor.bin"))
        // .include_file("mod.rs")
        .out_dir(proto_gen_dir)
        .compile(&[proto_file], &["proto"])
        .unwrap_or_else(|e| panic!("Failed to compile protos {:?}", e));

    Ok(())
}