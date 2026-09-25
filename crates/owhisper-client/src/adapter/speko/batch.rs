use std::path::{Path, PathBuf};

use owhisper_interface::ListenParams;
use owhisper_interface::batch::{Alternatives, Channel, Response as BatchResponse, Results, Word};
use reqwest::StatusCode;
use reqwest::multipart::{Form, Part};
use serde::Deserialize;

use super::{DEFAULT_API_BASE, SpekoAdapter};
use crate::adapter::meta::MetaAdapter;
use crate::adapter::meta::batch::encode_mono_wav;
use crate::adapter::{
    BatchFuture, BatchSttAdapter, ClientWithMiddleware, MIXED_CAPTURE_CHANNEL,
    append_path_if_missing,
};
use crate::error::Error;

// https://docs.speko.ai/relay/stt/batch
impl BatchSttAdapter for SpekoAdapter {
    fn provider_name(&self) -> &'static str {
        "speko"
    }

    fn is_supported_languages(
        &self,
        languages: &[anlg_language::Language],
        _model: Option<&str>,
    ) -> bool {
        Self::language_support_batch(languages).is_supported()
    }

    fn transcribe_file<'a, P: AsRef<Path> + Send + 'a>(
        &'a self,
        client: &'a ClientWithMiddleware,
        api_base: &'a str,
        api_key: &'a str,
        params: &'a ListenParams,
        file_path: P,
    ) -> BatchFuture<'a> {
        let path = file_path.as_ref().to_path_buf();
        Box::pin(do_transcribe_file(client, api_base, api_key, params, path))
    }
}

#[derive(Debug, Deserialize)]
struct SpekoSegment {
    text: String,
    start_ms: u64,
    end_ms: u64,
    #[serde(default)]
    speaker: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SpekoRoute {
    provider: String,
    model: String,
}

#[derive(Debug, Deserialize)]
struct SpekoUsage {
    duration_ms: u64,
}

#[derive(Debug, Deserialize)]
struct SpekoBatchResponse {
    #[serde(default)]
    text: String,
    #[serde(default)]
    segments: Vec<SpekoSegment>,
    #[serde(default)]
    route: Option<SpekoRoute>,
    #[serde(default)]
    usage: Option<SpekoUsage>,
}

const RETRY_DELAY: std::time::Duration = std::time::Duration::from_secs(2);

fn objective(model: Option<&str>) -> &'static str {
    match model {
        Some("auto-quality") => "quality",
        Some("auto-latency") => "latency",
        Some("auto-cost") => "cost",
        _ => "balanced",
    }
}

fn endpoint(api_base: &str) -> Result<url::Url, Error> {
    let mut url: url::Url = if api_base.is_empty() {
        DEFAULT_API_BASE
            .parse()
            .expect("invalid_default_speko_api_base")
    } else {
        api_base.parse().map_err(|e: url::ParseError| {
            Error::AudioProcessing(format!("invalid api_base: {e}"))
        })?
    };
    let path = url.path().trim_end_matches('/');
    if let Some(path) = path.strip_suffix("/v1").map(str::to_owned) {
        url.set_path(&path);
    }
    append_path_if_missing(&mut url, "v1/stt/transcriptions");
    Ok(url)
}

async fn do_transcribe_file(
    client: &ClientWithMiddleware,
    api_base: &str,
    api_key: &str,
    params: &ListenParams,
    file_path: PathBuf,
) -> Result<BatchResponse, Error> {
    let url = endpoint(api_base)?;
    let wav: bytes::Bytes = tokio::task::spawn_blocking(move || encode_mono_wav(&file_path))
        .await
        .map_err(|e| Error::AudioProcessing(e.to_string()))??
        .into();

    let mut request = serde_json::json!({
        "routing": { "mode": "auto", "objective": objective(params.model.as_deref()) },
        "options": { "diarization": true },
    });
    if let Some(language) = params.languages.first() {
        request["language"] = language.iso639().code().into();
    }

    let mut retried_unavailable = false;
    loop {
        match send(client, &url, api_key, &request, &wav).await {
            // Diarization narrows routing to the few models that support it; fall
            // back to an undiarized transcript rather than failing the request.
            Err(Error::UnexpectedStatus { status, body })
                if status == StatusCode::BAD_REQUEST
                    && body.contains("capability_unsupported")
                    && request["options"]["diarization"] == true =>
            {
                request["options"]["diarization"] = false.into();
            }
            // Speko marks transient upstream outages (e.g. 503 provider_unavailable)
            // as retryable even after its own failover is exhausted.
            Err(Error::UnexpectedStatus { status, body })
                if status.is_server_error()
                    && body.contains("\"retryable\":true")
                    && !retried_unavailable =>
            {
                retried_unavailable = true;
                tokio::time::sleep(RETRY_DELAY).await;
            }
            result => return result,
        }
    }
}

async fn send(
    client: &ClientWithMiddleware,
    url: &url::Url,
    api_key: &str,
    request: &serde_json::Value,
    wav: &bytes::Bytes,
) -> Result<BatchResponse, Error> {
    // Speko hashes parts in order for idempotency, so `request` must precede `audio`.
    let form = Form::new()
        .part(
            "request",
            Part::text(request.to_string())
                .mime_str("application/json")
                .expect("valid_mime"),
        )
        .part(
            "audio",
            Part::stream_with_length(wav.clone(), wav.len() as u64)
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .expect("valid_mime"),
        );

    let response = client
        .post(url.to_string())
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Idempotency-Key", uuid::Uuid::new_v4().to_string())
        .header("Accept", "application/json")
        .multipart(form)
        .send()
        .await?;

    let status = response.status();
    if status.is_success() {
        Ok(convert_response(response.json().await?))
    } else {
        Err(Error::UnexpectedStatus {
            status,
            body: crate::adapter::http::error_body(response).await,
        })
    }
}

fn convert_response(mut response: SpekoBatchResponse) -> BatchResponse {
    let mut words = Vec::new();
    let mut speaker_labels: Vec<String> = Vec::new();

    // Segments are optional in Speko's schema; keep a text-only transcript by
    // spreading it over the metered duration.
    if response.segments.is_empty() && !response.text.trim().is_empty() {
        response.segments.push(SpekoSegment {
            text: response.text.clone(),
            start_ms: 0,
            end_ms: response.usage.as_ref().map_or(0, |usage| usage.duration_ms),
            speaker: None,
        });
    }

    for segment in &response.segments {
        let speaker = segment
            .speaker
            .as_deref()
            .filter(|label| !label.is_empty())
            .map(|label| {
                speaker_labels
                    .iter()
                    .position(|known| known == label)
                    .unwrap_or_else(|| {
                        speaker_labels.push(label.to_string());
                        speaker_labels.len() - 1
                    })
            });
        let start = segment.start_ms as f64 / 1000.0;
        let end = segment.end_ms as f64 / 1000.0;
        for (token, start, end) in MetaAdapter::word_spans(&segment.text, start, end) {
            let normalized = token.trim_matches(|c: char| c.is_ascii_punctuation());
            words.push(Word {
                word: if normalized.is_empty() {
                    token.to_string()
                } else {
                    normalized.to_string()
                },
                start,
                end,
                confidence: 1.0,
                channel: if speaker.is_some() {
                    MIXED_CAPTURE_CHANNEL
                } else {
                    0
                },
                speaker,
                punctuated_word: Some(token.to_string()),
            });
        }
    }

    BatchResponse {
        metadata: serde_json::json!({
            "provider": "speko",
            "route_provider": response.route.as_ref().map(|route| &route.provider),
            "route_model": response.route.as_ref().map(|route| &route.model),
            "speaker_labels": speaker_labels,
            "timing_source": "provider_segment_interpolated",
        }),
        results: Results {
            channels: vec![Channel {
                alternatives: vec![Alternatives {
                    transcript: response.text.trim().to_string(),
                    confidence: 1.0,
                    words,
                }],
            }],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http_client::create_client;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    #[test]
    fn convert_response_indexes_speakers_and_spreads_words() {
        let response: SpekoBatchResponse = serde_json::from_value(serde_json::json!({
            "text": "Hello there. Hi.",
            "segments": [
                { "text": "Hello there.", "start_ms": 1000, "end_ms": 3000, "speaker": "1" },
                { "text": "Hi.", "start_ms": 3500, "end_ms": 4000, "speaker": "2" }
            ],
            "route": { "provider": "modulate", "model": "velma-2-stt-streaming" }
        }))
        .unwrap();

        let response = convert_response(response);

        let words = &response.results.channels[0].alternatives[0].words;
        assert_eq!(
            (words[0].word.as_str(), words[0].start, words[0].end),
            ("Hello", 1.0, 2.0)
        );
        assert_eq!(words[1].speaker, Some(0));
        assert_eq!(words[2].speaker, Some(1));
        assert_eq!(words[2].channel, MIXED_CAPTURE_CHANNEL);
        assert_eq!(response.metadata["route_provider"], "modulate");
    }

    #[test]
    fn maps_models_to_routing_objectives() {
        for (model, expected) in [
            (Some("auto-balanced"), "balanced"),
            (Some("auto-quality"), "quality"),
            (Some("auto-latency"), "latency"),
            (Some("auto-cost"), "cost"),
            (Some("nova-3"), "balanced"),
            (None, "balanced"),
        ] {
            assert_eq!(objective(model), expected, "{model:?}");
        }
    }

    #[test]
    fn builds_transcription_endpoint() {
        for (api_base, expected) in [
            ("", "https://router.speko.dev/v1/stt/transcriptions"),
            (
                "https://router.speko.dev/",
                "https://router.speko.dev/v1/stt/transcriptions",
            ),
            (
                "https://router.speko.dev/v1",
                "https://router.speko.dev/v1/stt/transcriptions",
            ),
            (
                "https://router.speko.dev/v1/",
                "https://router.speko.dev/v1/stt/transcriptions",
            ),
            (
                "https://eu.router.speko.dev/v1/stt/transcriptions",
                "https://eu.router.speko.dev/v1/stt/transcriptions",
            ),
        ] {
            assert_eq!(endpoint(api_base).unwrap().as_str(), expected);
        }
        assert!(matches!(
            endpoint("not a url"),
            Err(Error::AudioProcessing(message)) if message.starts_with("invalid api_base")
        ));
    }

    #[test]
    fn accepts_every_language() {
        assert_eq!(SpekoAdapter.provider_name(), "speko");
        assert!(SpekoAdapter.is_supported_languages(
            &[
                anlg_language::ISO639::Ko.into(),
                anlg_language::ISO639::En.into()
            ],
            Some("auto-balanced"),
        ));
    }

    #[test]
    fn keeps_punctuation_only_tokens_and_unlabeled_segments() {
        let response: SpekoBatchResponse = serde_json::from_value(serde_json::json!({
            "text": " Well ... ok ",
            "segments": [{ "text": "Well ... ok", "start_ms": 0, "end_ms": 3000, "speaker": "" }]
        }))
        .unwrap();

        let response = convert_response(response);

        let alternative = &response.results.channels[0].alternatives[0];
        assert_eq!(alternative.transcript, "Well ... ok");
        assert_eq!(alternative.words[1].word, "...");
        assert!(alternative.words.iter().all(|word| word.speaker.is_none()));
        assert!(alternative.words.iter().all(|word| word.channel == 0));
        assert!(response.metadata["route_provider"].is_null());
    }

    #[test]
    fn keeps_text_only_transcripts() {
        let response: SpekoBatchResponse = serde_json::from_value(serde_json::json!({
            "text": "Hello there.",
            "usage": { "duration_ms": 2000 }
        }))
        .unwrap();

        let words = convert_response(response).results.channels[0].alternatives[0]
            .words
            .clone();

        assert_eq!(words.len(), 2);
        assert_eq!((words[0].start, words[1].end), (0.0, 2.0));
    }

    #[tokio::test]
    async fn surfaces_request_and_response_failures() {
        let transcribe = |api_base: String, path: &'static str| async move {
            SpekoAdapter
                .transcribe_file(
                    &create_client(),
                    &api_base,
                    "test-key",
                    &ListenParams::default(),
                    path,
                )
                .await
                .unwrap_err()
        };
        let audio = anlg_data::english_1::AUDIO_PATH;

        assert!(matches!(
            transcribe("not a url".to_string(), audio).await,
            Error::AudioProcessing(_)
        ));
        assert!(matches!(
            transcribe("http://127.0.0.1:1".to_string(), "/nonexistent.wav").await,
            Error::AudioProcessing(_)
        ));
        assert!(!matches!(
            transcribe("http://127.0.0.1:1".to_string(), audio).await,
            Error::UnexpectedStatus { .. } | Error::AudioProcessing(_)
        ));

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&server)
            .await;
        assert!(!matches!(
            transcribe(server.uri(), audio).await,
            Error::UnexpectedStatus { .. } | Error::AudioProcessing(_)
        ));
    }

    #[tokio::test]
    async fn does_not_retry_non_retryable_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "error": { "code": "authentication_failed", "retryable": false }
            })))
            .mount(&server)
            .await;

        let error = SpekoAdapter
            .transcribe_file(
                &create_client(),
                &server.uri(),
                "bad-key",
                &ListenParams::default(),
                anlg_data::english_1::AUDIO_PATH,
            )
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            Error::UnexpectedStatus { status, .. } if status == StatusCode::UNAUTHORIZED
        ));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn retries_without_diarization_when_capability_is_unsupported() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/stt/transcriptions"))
            .respond_with(|request: &Request| {
                if String::from_utf8_lossy(&request.body).contains("\"diarization\":true") {
                    ResponseTemplate::new(400).set_body_json(serde_json::json!({
                        "error": { "code": "capability_unsupported", "message": "no routable model" }
                    }))
                } else {
                    ResponseTemplate::new(200).set_body_json(serde_json::json!({
                        "text": "안녕하세요",
                        "segments": [{ "text": "안녕하세요", "start_ms": 0, "end_ms": 1000 }]
                    }))
                }
            })
            .mount(&server)
            .await;

        let response = SpekoAdapter
            .transcribe_file(
                &create_client(),
                &server.uri(),
                "test-key",
                &ListenParams {
                    model: Some("auto-quality".to_string()),
                    languages: vec![anlg_language::ISO639::Ko.into()],
                    ..Default::default()
                },
                anlg_data::english_1::AUDIO_PATH,
            )
            .await
            .unwrap();

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 2);
        let first = String::from_utf8_lossy(&requests[0].body);
        assert!(first.contains("\"objective\":\"quality\""));
        assert!(first.contains("\"language\":\"ko\""));
        assert!(first.find("name=\"request\"") < first.find("name=\"audio\""));
        assert_ne!(
            requests[0].headers.get("idempotency-key"),
            requests[1].headers.get("idempotency-key")
        );
        assert_eq!(
            response.results.channels[0].alternatives[0].transcript,
            "안녕하세요"
        );
    }

    #[tokio::test]
    async fn retries_once_when_provider_is_temporarily_unavailable() {
        let server = MockServer::start().await;
        let unavailable = serde_json::json!({
            "error": { "code": "provider_unavailable", "retryable": true }
        });
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503).set_body_json(&unavailable))
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "Hello.",
                "segments": [{ "text": "Hello.", "start_ms": 0, "end_ms": 500 }]
            })))
            .mount(&server)
            .await;

        let response = SpekoAdapter
            .transcribe_file(
                &create_client(),
                &server.uri(),
                "test-key",
                &ListenParams::default(),
                anlg_data::english_1::AUDIO_PATH,
            )
            .await
            .unwrap();

        assert_eq!(server.received_requests().await.unwrap().len(), 2);
        assert_eq!(
            response.results.channels[0].alternatives[0].transcript,
            "Hello."
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_batch_file() {
        let (language, audio) = match std::env::var("SPEKO_LANGUAGE").as_deref() {
            Ok("ko") => (anlg_language::ISO639::Ko, "korean_1"),
            _ => (anlg_language::ISO639::En, "english_1"),
        };
        let response = SpekoAdapter
            .transcribe_file(
                &create_client(),
                "",
                &std::env::var("SPEKO_API_KEY").expect("SPEKO_API_KEY not set"),
                &ListenParams {
                    languages: vec![language.into()],
                    ..Default::default()
                },
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join(format!("../../crates/data/src/{audio}/audio.wav")),
            )
            .await
            .unwrap();
        println!(
            "{}",
            serde_json::to_string_pretty(&response.metadata).unwrap()
        );
        println!(
            "{}",
            response.results.channels[0].alternatives[0].transcript
        );
    }
}
