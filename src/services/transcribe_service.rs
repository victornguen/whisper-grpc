use tonic::{Request, Response, Status};
use crate::pb::transcribe_v1::{Audio, Phrase, Phrases};
use crate::pb::transcribe_v1::transcribe_service_server::TranscribeService;
use crate::transcribe::transcriber::Transcriber;

#[derive(Debug, Default)]
pub struct Service {
    transcriber: Transcriber
}

#[tonic::async_trait]
impl TranscribeService for Service {
    async fn transcribe(&self, request: Request<Audio>) -> Result<Response<Phrases>, Status> {
        Ok(Response::new(Phrases {
            phrases: vec![Phrase{
                text: "Hello World".to_string(),
                word_times: vec![]
            }]
        }))
    }

}

// impl Default for Service {
//     fn default() -> Self {
//         Self {
//             transcriber: Transcriber::default()
//         }
//     }
// }