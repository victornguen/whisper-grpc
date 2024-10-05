use anyhow::Context;
use cuda_driver_sys::{cuDeviceGetCount, cudaError_enum};
use serde::de::Error;
use whisper_rs::{FullParams, WhisperContext, WhisperContextParameters, WhisperError, WhisperState, WhisperTokenData};
use crate::pb::transcribe_v1::Word;
use crate::transcribe::transcriber::errors::TranscribeError;

#[derive(Debug)]
pub struct Transcriber {
    ctx: WhisperContext,
    threshold_interval: i64,
}

impl Transcriber {
    pub fn new(threshold_interval: i64) -> Transcriber {
        let model_path = env!("WHISPER_MODEL_PATH", "/opt/whisper/ggml-large-v3.bin");
        let mut context_param = WhisperContextParameters::default();

        context_param.use_gpu(Self::cuda_available());

        let ctx = WhisperContext::new_with_params(
            model_path,
            context_param,
        ).expect("failed to load model");

        Self {
            ctx,
            threshold_interval,
        }
    }

    pub fn transcribe<'a>(&self, audio: &[f32], params: FullParams) -> Result<Vec<Word>, TranscribeError> {
        use anyhow::Result;
        use anyhow::Context;
        use anyhow;
        let mut state = self.ctx.create_state().expect("failed to create state");
        state
            .full(params, audio)
            .expect("failed to run model");
        let words = self.get_words(&state).context("get words")?;
        let phrases = self.get_phrases(&words).context("get phrases")?;
        Ok(())
    }

    fn get_words(&self, state: &WhisperState) -> Result<Vec<Word>, TranscribeError> {
        let num_segments = state
            .full_n_segments()
            .context("get number of segments")
            .map_err(|cause| TranscribeError { cause })?;
        let words: Vec<Word> = (0..num_segments).map(|seg_number| {
            let segment = state
                .full_get_segment_text(seg_number)
                .context("get segment")?;
            let start_timestamp = state
                .full_get_segment_t0(seg_number)
                .context("get segment start timestamp")?;
            let end_timestamp = state
                .full_get_segment_t1(seg_number)
                .context("get segment end timestamp")?;
            Ok(Word {
                word: segment,
                start_ms: start_timestamp,
                end_ms: end_timestamp,
            })
        }
        ).collect();

        Ok(words)
    }

    fn get_phrases(&self, words: &Vec<Word>) -> Result<Vec<Word>, TranscribeError> {
        let mut phrases = Vec::new();
        let mut current_phrase = Vec::new();
        let mut current_phrase_start = 0;
        let mut current_phrase_end = 0;
        for word in words {
            if word.start_ms - current_phrase_end > self.threshold_interval {
                if !current_phrase.is_empty() {
                    phrases.push(Word {
                        word: current_phrase.join(" "),
                        start_ms: current_phrase_start,
                        end_ms: current_phrase_end,
                    });
                    current_phrase.clear();
                }
                current_phrase_start = word.start_ms;
            }
            current_phrase.push(word.word.clone());
            current_phrase_end = word.end_ms;
        }
        if !current_phrase.is_empty() {
            phrases.push(Word {
                word: current_phrase.join(" "),
                start_ms: current_phrase_start,
                end_ms: current_phrase_end,
            });
        }
        Ok(phrases)
    }

    fn cuda_available() -> bool {
        unsafe {
            let mut device_count: i32 = 0;
            let result = cuDeviceGetCount(&mut device_count as *mut i32);
            result == cudaError_enum::CUDA_SUCCESS && device_count > 0
        }
    }
}

impl Default for Transcriber {
    fn default() -> Self {
        Self::new(500)
    }
}

mod errors {
    use thiserror::Error;

    #[derive(Error, Debug)]
    #[error("failed to get transcription")]
    pub struct TranscribeError {
        pub cause: anyhow::Error,
    }

}