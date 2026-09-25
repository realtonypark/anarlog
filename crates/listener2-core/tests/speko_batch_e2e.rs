use std::sync::{Arc, Mutex};

use listener2_core::{BatchEvent, BatchParams, BatchProvider, BatchRuntime, run_batch};

#[derive(Default)]
struct RecordingRuntime(Mutex<Vec<BatchEvent>>);

impl BatchRuntime for RecordingRuntime {
    fn emit(&self, event: BatchEvent) {
        self.0.lock().unwrap().push(event);
    }
}

// Runs the same path as the desktop "After recording" flow against the live
// Speko Router: SPEKO_API_KEY=... [SPEKO_LANGUAGE=ko] [SPEKO_MODEL=auto-balanced]
// cargo test -p listener2-core
// --test speko_batch_e2e -- --ignored --nocapture
// The audio language must match the language saved on the key.
#[ignore]
#[tokio::test]
async fn speko_batch_transcribes() {
    let (language, audio) = match std::env::var("SPEKO_LANGUAGE").as_deref() {
        Ok("ko") => (anlg_language::ISO639::Ko, "korean_1"),
        _ => (anlg_language::ISO639::En, "english_1"),
    };
    let model = std::env::var("SPEKO_MODEL").unwrap_or_else(|_| "key-settings".to_string());
    let runtime = Arc::new(RecordingRuntime::default());

    let output = run_batch(
        runtime.clone(),
        BatchParams {
            session_id: "speko-e2e".to_string(),
            provider: BatchProvider::Speko,
            file_path: format!(
                "{}/../data/src/{audio}/audio.wav",
                env!("CARGO_MANIFEST_DIR")
            ),
            model: Some(model.clone()),
            base_url: "https://router.speko.dev".to_string(),
            api_key: std::env::var("SPEKO_API_KEY").expect("SPEKO_API_KEY not set"),
            languages: vec![language.into()],
            keywords: vec![],
            num_speakers: None,
            min_speakers: None,
            max_speakers: None,
            known_speakers: vec![],
        },
    )
    .await
    .unwrap();

    let alternative = &output.response.results.channels[0].alternatives[0];
    println!("{}", output.response.metadata);
    println!("{}", alternative.transcript);
    assert!(!alternative.transcript.trim().is_empty());
    if model.starts_with("auto-") {
        assert!(alternative.words.iter().any(|word| word.speaker.is_some()));
    }
    assert!(
        runtime
            .0
            .lock()
            .unwrap()
            .iter()
            .any(|event| matches!(event, BatchEvent::BatchCompleted { .. }))
    );
}
