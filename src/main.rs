use hound;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperTokenData};

fn main() {

    // Load a context and model
    // let model_path = "D:/DownloadsD/ggml-small.bin";
    let model_path = "D:/DownloadsD/ggml-large-v3.bin";
    let mut context_param = WhisperContextParameters::default();
    context_param.use_gpu(true);
    let ctx = WhisperContext::new_with_params(
        model_path,
        context_param,
    ).expect("failed to load model");

    // create a params object
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    // let mut params = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 1, patience: 0.5 });
    params.set_language(None);
    params.set_max_len(1);
    params.set_token_timestamps(true);
    // params.set_single_segment(true);
    params.set_split_on_word(true);
    // params.set_split_on_word(true);
    // params.set_single_segment(true);
    // params.set_tdrz_enable(true);

    // Load and preprocess the audio
    let audio_path = "D:/DocumentsD/testAudio/out_stereo_ch1_16k.wav";
    // let audio_path = "D:/DocumentsD/testAudio/37048.wav";
    let audio = load_wav(audio_path);

    // now we can run the model
    let mut state = ctx.create_state().expect("failed to create state");
    state
        .full(params, &audio[..])
        .expect("failed to run model");

    // fetch the results
    let num_segments = state
        .full_n_segments()
        .expect("failed to get number of segments");
    for i in 0..num_segments {
        let segment = state
            .full_get_segment_text(i)
            .expect("failed to get segment");
        let num_tokens = state
            .full_n_tokens(i)
            .expect("failed to get number of tokens");
        let start_timestamp = state
            .full_get_segment_t0(i)
            .expect("failed to get segment start timestamp");
        let end_timestamp = state
            .full_get_segment_t1(i)
            .expect("failed to get segment end timestamp");
        let tokens_data: Vec<(WhisperTokenData, String)> = (0..num_tokens).map(|token_num| {
            let data = state
                .full_get_token_data(i, token_num)
                .expect("failed to get token data");
            let word = state.full_get_token_text_lossy(i, token_num).expect("failed to get token text");
            (data, word)
        })
            .collect();
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
        // tokens_data.iter().for_each(|(data, word)| {
        //     println!("  [{} - {}]: {}", data.t0, data.t1, word)
        // })
    }

    // Transcribe the audio
    // transcribe_audio(&ctx, &audio);
}

fn load_wav(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("Failed to open WAV file");
    let samples: Vec<f32> = reader.samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();
    samples
}

// fn transcribe_audio(ctx: &WhisperContext, audio: &[f32]) {
//     let params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
//
//     ctx.full(params, audio).expect("Failed to process audio");
//
//     let num_segments = ctx.full_n_segments();
//     for i in 0..num_segments {
//         let segment = ctx.full_get_segment_text(i).expect("Failed to get segment");
//         println!("{}", segment);
//     }
// }