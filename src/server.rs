use clap::{Arg, Command};
use tonic::transport::Server;
use crate::pb::transcribe_v1::transcribe_service_server::TranscribeServiceServer;
use crate::services::transcribe_service;
use crate::settings::settings::Settings;

mod services;
mod transcribe;
mod settings;
mod pb;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let command = Command::new("Speech-to-Text gRPC service")
        .version("1.0")
        .about("Speech-to-text microservice with Whisper written in Rust")
        .arg(Arg::new("config")
            .short('c')
            .long("config")
            .help("Configuration file location")
            .default_value("config.yaml"));

    let matches = command.get_matches();
    let config_location = matches.get_one::<String>("config").unwrap_or(&"".to_string()).to_string();
    let settings = Settings::new(&config_location, "TRANSCRIBE")?;

    let transcribe_service = transcribe_service::Service::default();

    let addr = format!("{}:{}", settings.server.host, settings.server.port).parse()?;
    println!("Server listening on {}", addr);

    Server::builder()
        .add_service(TranscribeServiceServer::new(transcribe_service))
        .serve(addr)
        .await?;

    Ok(())
}