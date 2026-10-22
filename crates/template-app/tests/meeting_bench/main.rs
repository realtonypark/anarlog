//! Meeting-notes benchmark: 8 candidates x N meetings x 3 judges.
//!
//! Phase 1 renders the production enhance prompt for every meeting and runs
//! each candidate model. Phase 2 grades every note with three System One
//! decision judges (mammoth, jev, decider) on the shared `JUDGE_QUESTIONS`
//! rubric. Phase 3 asks each judge to pick the better note for every pair
//! (randomized order, fixed seed); the final ranking is Borda over the three
//! per-judge rankings. Results append to run.jsonl (crash-safe, reruns
//! skip finished items). `report` renders report.md + index.html from it.
//!
//! Usage:
//!   meeting_bench run    # needs MEETING_BENCH_CORPUS + MEETING_BENCH_RUN_DIR
//!   meeting_bench report # needs MEETING_BENCH_RUN_DIR (+ MEETING_BENCH_OUT_DIR)
//!
//! Env: MEETING_BENCH_KEYS (default ~/.config/anarlog-bench) holds
//! openrouter.key, ramp.key, baseten.key. Optional filters:
//! MEETING_BENCH_MEETINGS, MEETING_BENCH_MODELS, MEETING_BENCH_JUDGES
//! (comma lists), MEETING_BENCH_CONCURRENCY (default 4).

mod report;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anlg_openrouter::{
    ChatCompletionRequest, ChatMessage, Client as ChatClient, Effort, ReasoningConfig, Role,
};
use anlg_template_eval::{BROWSER_USER_AGENT, SystemOneClient};
use template_app::{
    EnhanceSystem, EnhanceUser, Participant, Segment, Session, Template, Transcript, render,
};

const RAMP_BASE_URL: &str = "https://api.router.com/v1";
const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
const MAMMOTH_SYSTEMONE_URL: &str = "https://inference.baseten.co/v1/systemone";
const JEV_SYSTEMONE_URL: &str = "https://api.router.com/v1/systemone";
const DECIDER_SYSTEMONE_URL: &str = "https://openrouter.ai/api/alpha/decisions";
// Production omits the cap (Anthropic: 64k); 8k truncated notes under high
// reasoning effort, so the bench caps far above any real note instead.
const GEN_MAX_TOKENS: u32 = 32000;
// Every judge sees identical inputs. The state budget fits the smallest judge
// window (Baseten mammoth: 16k tokens, never truncated server-side).
const JUDGE_STATE_BUDGET_CHARS: usize = 26_000;
const JUDGE_PROMPT_HEADROOM_CHARS: usize = 1_000;
const JUDGE_NOTE_CAP_CHARS: usize = 12_000;
const REQUEST_TIMEOUT_SECS: u64 = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatProvider {
    Ramp,
    OpenRouter,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    id: &'static str,
    provider: ChatProvider,
}

// Fixed order doubles as blind labels A..H for judges.
const CANDIDATES: [Candidate; 8] = [
    Candidate {
        id: "claude-haiku-5-5",
        provider: ChatProvider::Ramp,
    },
    Candidate {
        id: "gpt-6-luna",
        provider: ChatProvider::Ramp,
    },
    Candidate {
        id: "deepseek-v4.1-flash-1017",
        provider: ChatProvider::Ramp,
    },
    Candidate {
        id: "glm-5p3-flash",
        provider: ChatProvider::Ramp,
    },
    Candidate {
        id: "grok-4.6",
        provider: ChatProvider::Ramp,
    },
    Candidate {
        id: "google/gemini-3.8-flash",
        provider: ChatProvider::OpenRouter,
    },
    Candidate {
        id: "nvidia/nemotron-3.5-super-vl",
        provider: ChatProvider::OpenRouter,
    },
    Candidate {
        id: "meta/muse-spark-1.3-contributor",
        provider: ChatProvider::OpenRouter,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Judge {
    Mammoth,
    Jev,
    Decider,
}

impl Judge {
    fn id(self) -> &'static str {
        match self {
            Judge::Mammoth => "mammoth",
            Judge::Jev => "jev",
            Judge::Decider => "decider",
        }
    }

    fn model(self) -> &'static str {
        match self {
            Judge::Mammoth => "decision-models/baseten-mammoth-27b-latest",
            Judge::Jev => "jev-1.14.0",
            Judge::Decider => "perplexity/pplx-decider-v1.2",
        }
    }
}

const JUDGES: [Judge; 3] = [Judge::Mammoth, Judge::Jev, Judge::Decider];

#[derive(Debug, Clone, serde::Deserialize)]
struct CorpusSegment {
    speaker: String,
    text: String,
    #[allow(dead_code)]
    start_ms: Option<i64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct CorpusParticipant {
    name: String,
    job_title: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct CorpusMeeting {
    id: String,
    title: String,
    #[allow(dead_code)]
    started_at: String,
    #[allow(dead_code)]
    ended_at: String,
    language: String,
    participants: Vec<CorpusParticipant>,
    segments: Vec<CorpusSegment>,
    #[allow(dead_code)]
    transcript_chars: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Corpus {
    meetings: Vec<CorpusMeeting>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
struct GenRecord {
    #[serde(rename = "t")]
    kind: String,
    meeting: String,
    model: String,
    blind: String,
    output: String,
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: Option<f64>,
    elapsed_ms: u128,
    prompt_hash: String,
}

#[derive(Debug, Clone, serde::Serialize)]
#[allow(dead_code)]
struct JudgeRecord {
    #[serde(rename = "t")]
    kind: String,
    meeting: String,
    model: String,
    judge: String,
    scores: HashMap<String, f64>,
    evidence: HashMap<String, String>,
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: Option<f64>,
    elapsed_ms: u128,
    truncated: bool,
    transcript_chars: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
#[allow(dead_code)]
struct MetaRecord {
    #[serde(rename = "t")]
    kind: String,
    started_at_ms: u128,
    candidates: Vec<String>,
    judges: Vec<String>,
    meetings: Vec<String>,
    blind_labels: HashMap<String, String>,
    effort: String,
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_list(key: &str) -> Option<HashSet<String>> {
    std::env::var(key).ok().map(|value| {
        value
            .split(',')
            .map(|part| part.trim().to_string())
            .filter(|part| !part.is_empty())
            .collect()
    })
}

fn read_key(keys_dir: &str, name: &str) -> Result<String, String> {
    let path = format!("{keys_dir}/{name}");
    std::fs::read_to_string(&path)
        .map(|key| key.trim().to_string())
        .map_err(|err| format!("failed to read {path}: {err}"))
}

fn fnv1a_hex(text: &str) -> String {
    // FNV-1a stand-in: identifies prompt versions, not security.
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a:{hash:016x}")
}

fn default_keys_dir() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    format!("{home}/.config/anarlog-bench")
}

fn render_prompts(meeting: &CorpusMeeting) -> Result<(String, String), String> {
    let language = (!meeting.language.trim().is_empty()).then(|| meeting.language.clone());
    let system = render(Template::EnhanceSystem(EnhanceSystem {
        language,
        format_override: String::new(),
    }))
    .map_err(|err| format!("system render failed: {err}"))?;
    let user = render(Template::EnhanceUser(Box::new(EnhanceUser {
        session: Session {
            title: Some(meeting.title.clone()),
            started_at: None,
            ended_at: None,
            event: None,
        },
        participants: meeting
            .participants
            .iter()
            .map(|p| Participant {
                name: p.name.clone(),
                job_title: p.job_title.clone(),
            })
            .collect(),
        template: None,
        transcripts: vec![Transcript {
            segments: meeting
                .segments
                .iter()
                .map(|s| Segment {
                    text: s.text.clone(),
                    speaker: s.speaker.clone(),
                    start_ms: s.start_ms,
                })
                .collect(),
            started_at: None,
            ended_at: None,
            now_ms: None,
        }],
        pre_meeting_memo: String::new(),
        post_meeting_memo: String::new(),
    })))
    .map_err(|err| format!("user render failed: {err}"))?;
    Ok((system, user))
}

fn transcript_text(meeting: &CorpusMeeting) -> String {
    let mut out = String::new();
    for segment in &meeting.segments {
        out.push_str(&segment.speaker);
        out.push_str(": ");
        out.push_str(segment.text.trim());
        out.push('\n');
    }
    out
}

fn clip(text: &str, cap_chars: usize, marker: &str) -> (String, bool) {
    if text.chars().count() <= cap_chars {
        return (text.to_string(), false);
    }
    let mut clipped = text.chars().take(cap_chars).collect::<String>();
    clipped.push_str(marker);
    (clipped, true)
}

/// Fit transcript + notes into the shared state budget; returns the clipped
/// transcript and whether anything was clipped.
fn fit_judge_input(transcript: &str, notes: &[&str]) -> (String, bool, usize) {
    let orig_chars = transcript.chars().count();
    let notes_len: usize = notes
        .iter()
        .map(|note| note.chars().count().min(JUDGE_NOTE_CAP_CHARS))
        .sum();
    let transcript_cap = JUDGE_STATE_BUDGET_CHARS
        .saturating_sub(notes_len)
        .saturating_sub(JUDGE_PROMPT_HEADROOM_CHARS)
        .max(4_000);
    let (clipped, truncated) = clip(
        transcript,
        transcript_cap,
        "\n[...transcript clipped to fit the judge window...]",
    );
    (clipped, truncated, orig_chars)
}

fn fit_note(note: &str) -> String {
    clip(note, JUDGE_NOTE_CAP_CHARS, "\n[...note clipped...]").0
}

fn build_chat_client(base_url: &str, api_key: &str) -> Result<ChatClient, String> {
    let http = reqwest::Client::builder()
        .user_agent(BROWSER_USER_AGENT)
        .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .build()
        .map_err(|err| format!("failed to build http client: {err}"))?;
    Ok(ChatClient::new(api_key.to_string())
        .with_base_url(base_url.to_string())
        .with_http_client(http))
}

fn is_retryable(error: &anlg_openrouter::Error) -> bool {
    match error {
        anlg_openrouter::Error::Api { status, .. } => *status == 429 || *status >= 500,
        anlg_openrouter::Error::Http(_) => true,
        _ => false,
    }
}

struct Clients {
    ramp: ChatClient,
    openrouter: ChatClient,
    mammoth: SystemOneClient,
    jev: SystemOneClient,
    decider: SystemOneClient,
}

impl Clients {
    fn judge_client(&self, judge: Judge) -> &SystemOneClient {
        match judge {
            Judge::Mammoth => &self.mammoth,
            Judge::Jev => &self.jev,
            Judge::Decider => &self.decider,
        }
    }
}

impl Clients {
    fn chat(&self, provider: ChatProvider) -> &ChatClient {
        match provider {
            ChatProvider::Ramp => &self.ramp,
            ChatProvider::OpenRouter => &self.openrouter,
        }
    }
}

fn build_clients(keys_dir: &str) -> Result<Clients, String> {
    Ok(Clients {
        ramp: build_chat_client(RAMP_BASE_URL, &read_key(keys_dir, "ramp.key")?)?,
        openrouter: build_chat_client(OPENROUTER_BASE_URL, &read_key(keys_dir, "openrouter.key")?)?,
        mammoth: SystemOneClient::new(MAMMOTH_SYSTEMONE_URL, read_key(keys_dir, "baseten.key")?)?,
        jev: SystemOneClient::new(JEV_SYSTEMONE_URL, read_key(keys_dir, "ramp.key")?)?,
        decider: SystemOneClient::new(
            DECIDER_SYSTEMONE_URL,
            read_key(keys_dir, "openrouter.key")?,
        )?,
    })
}

async fn chat_with_retry(
    client: &ChatClient,
    request: &ChatCompletionRequest,
) -> Result<anlg_openrouter::ChatCompletionResponse, String> {
    let mut attempt = 0;
    loop {
        attempt += 1;
        match client.chat_completion(request).await {
            Ok(response) => return Ok(response),
            Err(err) => {
                if !is_retryable(&err) || attempt >= 3 {
                    return Err(format!("chat failed: {err}"));
                }
                tokio::time::sleep(Duration::from_millis(1000 * attempt as u64)).await;
            }
        }
    }
}

fn response_text(response: &anlg_openrouter::ChatCompletionResponse) -> Result<String, String> {
    let content = response
        .choices
        .first()
        .and_then(|choice| choice.message.content.as_ref())
        .ok_or_else(|| "chat response carried no content".to_string())?;
    if let Some(text) = content.as_text() {
        return Ok(text.to_string());
    }
    if let anlg_openrouter::Content::Parts(parts) = content {
        let joined = parts
            .iter()
            .filter_map(|part| match part {
                anlg_openrouter::ContentPart::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !joined.trim().is_empty() {
            return Ok(joined);
        }
    }
    Err("chat response carried no text".to_string())
}

/// Reasoning effort mirror of apps/desktop/src/ai/reasoning-effort.ts: the app
/// sends raw `reasoning_effort` down the OpenAI path (Ramp Router here) and
/// `reasoning.effort` down the OpenRouter path. "default" omits both.
fn effort_from_env() -> Option<String> {
    match env_or("MEETING_BENCH_EFFORT", "high")
        .to_lowercase()
        .as_str()
    {
        "xhigh" | "high" | "medium" | "low" | "minimal" => {
            Some(env_or("MEETING_BENCH_EFFORT", "high").to_lowercase())
        }
        _ => None,
    }
}

fn effort_enum(effort: &str) -> Option<Effort> {
    match effort {
        "xhigh" => Some(Effort::Xhigh),
        "high" => Some(Effort::High),
        "medium" => Some(Effort::Medium),
        "low" => Some(Effort::Low),
        "minimal" => Some(Effort::Minimal),
        _ => None,
    }
}

async fn generate_note(
    clients: &Clients,
    candidate: Candidate,
    system: &str,
    user: &str,
    effort: Option<&str>,
) -> Result<(String, u64, u64, Option<f64>), String> {
    let mut request = ChatCompletionRequest {
        model: Some(candidate.id.to_string()),
        messages: vec![
            ChatMessage::new(Role::System, system.to_string()),
            ChatMessage::new(Role::User, user.to_string()),
        ],
        max_tokens: Some(GEN_MAX_TOKENS),
        temperature: None,
        ..Default::default()
    };
    if let Some(effort) = effort {
        match candidate.provider {
            ChatProvider::Ramp => request.reasoning_effort = Some(effort.to_string()),
            ChatProvider::OpenRouter => {
                request.reasoning = Some(ReasoningConfig {
                    effort: effort_enum(effort),
                    summary: None,
                })
            }
        }
    }
    let client = clients.chat(candidate.provider);
    let response = chat_with_retry(client, &request).await?;
    let output = response_text(&response)?;
    let (input_tokens, output_tokens) = response
        .usage
        .as_ref()
        .map(|usage| (usage.prompt_tokens as u64, usage.completion_tokens as u64))
        .unwrap_or((0, 0));
    // Costs are computed in the report from tokens x list price; the
    // provider-reported figure is never used.
    Ok((output, input_tokens, output_tokens, None))
}

fn judge_state(blind: &str, transcript: &str, note: &str) -> String {
    format!(
        "A meeting note from Candidate {blind} is under review. Grade it against the transcript.\n\n# Transcript\n{transcript}\n\n# Meeting note under review\n{note}"
    )
}

async fn judge_note(
    clients: &Clients,
    judge: Judge,
    blind: &str,
    transcript: &str,
    note: &str,
) -> Result<(HashMap<String, f64>, u64, u64, Option<f64>), String> {
    let client = clients.judge_client(judge);
    let state = judge_state(blind, transcript, note);
    let response = client.judge_state(judge.model(), &state).await?;
    let (input_tokens, output_tokens) = response
        .usage
        .as_ref()
        .map(|usage| (usage.input_tokens, usage.output_tokens))
        .unwrap_or((0, 0));
    let cost_usd = None;
    let scores = anlg_template_eval::noul_scores(&response);
    let missing: Vec<_> = anlg_template_eval::JUDGE_QUESTIONS
        .iter()
        .map(|(id, _)| *id)
        .filter(|id| !scores.contains_key(*id))
        .collect();
    if !missing.is_empty() {
        return Err(format!("judge skipped criteria: {}", missing.join(",")));
    }
    Ok((scores, input_tokens, output_tokens, cost_usd))
}

#[derive(Debug, Clone, serde::Serialize)]
#[allow(dead_code)]
struct PairRecord {
    #[serde(rename = "t")]
    kind: String,
    meeting: String,
    judge: String,
    model_a: String,
    model_b: String,
    first: String,
    p_first: f64,
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: Option<f64>,
    elapsed_ms: u128,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct JsonlProbe {
    #[serde(rename = "t")]
    kind: String,
    #[serde(default)]
    meeting: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    judge: String,
    #[serde(default)]
    model_a: String,
    #[serde(default)]
    model_b: String,
}

/// Deterministic per-pair order from the pair key: resume-proof, no rng state.
fn pair_swapped(meeting: &str, judge: &str, model_a: &str, model_b: &str) -> bool {
    let hash = fnv1a_hex(&format!("{meeting}\n{judge}\n{model_a}\n{model_b}"));
    hash.chars()
        .last()
        .and_then(|c| c.to_digit(16))
        .is_some_and(|digit| digit % 2 == 1)
}

fn pair_state(
    blind_first: &str,
    note_first: &str,
    blind_second: &str,
    note_second: &str,
    transcript: &str,
) -> String {
    format!(
        "Two meeting notes (Candidate {blind_first} and Candidate {blind_second}) are under review. Pick the better one against the transcript.\n\n# Transcript\n{transcript}\n\n# Note A (Candidate {blind_first})\n{note_first}\n\n# Note B (Candidate {blind_second})\n{note_second}"
    )
}

/// Append-only JSONL store; reruns skip finished (meeting, model[, judge]).
struct Store {
    path: String,
    done_gens: HashSet<(String, String)>,
    done_judges: HashSet<(String, String, String)>,
    done_pairs: HashSet<(String, String, String, String)>,
}

impl Store {
    fn open(run_dir: &str) -> Result<Self, String> {
        std::fs::create_dir_all(run_dir)
            .map_err(|err| format!("failed to create {run_dir}: {err}"))?;
        let path = format!("{run_dir}/run.jsonl");
        let mut store = Self {
            path,
            done_gens: HashSet::new(),
            done_judges: HashSet::new(),
            done_pairs: HashSet::new(),
        };
        if let Ok(text) = std::fs::read_to_string(&store.path) {
            for line in text.lines() {
                let Ok(probe) = serde_json::from_str::<JsonlProbe>(line) else {
                    continue;
                };
                match probe.kind.as_str() {
                    "gen" => {
                        store.done_gens.insert((probe.meeting, probe.model));
                    }
                    "judge" => {
                        store
                            .done_judges
                            .insert((probe.meeting, probe.model, probe.judge));
                    }
                    "pair" => {
                        let (mut a, mut b) = (probe.model_a, probe.model_b);
                        if a > b {
                            std::mem::swap(&mut a, &mut b);
                        }
                        store.done_pairs.insert((probe.meeting, probe.judge, a, b));
                    }
                    _ => {}
                }
            }
        }
        Ok(store)
    }

    fn append(&self, record: &impl serde::Serialize) -> Result<(), String> {
        use std::io::Write;
        let mut line = serde_json::to_vec(record).map_err(|err| format!("encode: {err}"))?;
        line.push(b'\n');
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|err| format!("open {}: {err}", self.path))?;
        file.write_all(&line)
            .map_err(|err| format!("write {}: {err}", self.path))?;
        file.sync_all().map_err(|err| format!("sync: {err}"))?;
        Ok(())
    }
}

fn blind_label(index: usize) -> String {
    ((b'A' + index as u8) as char).to_string()
}

async fn run() -> Result<(), String> {
    let corpus_path = std::env::var("MEETING_BENCH_CORPUS")
        .map_err(|_| "set MEETING_BENCH_CORPUS".to_string())?;
    let run_dir = std::env::var("MEETING_BENCH_RUN_DIR")
        .map_err(|_| "set MEETING_BENCH_RUN_DIR".to_string())?;
    let keys_dir = env_or("MEETING_BENCH_KEYS", &default_keys_dir());
    let concurrency: usize = env_or("MEETING_BENCH_CONCURRENCY", "4")
        .parse()
        .unwrap_or(4)
        .max(1);
    let meeting_filter = env_list("MEETING_BENCH_MEETINGS");
    let model_filter = env_list("MEETING_BENCH_MODELS");
    let judge_filter = env_list("MEETING_BENCH_JUDGES");

    let corpus_text = std::fs::read_to_string(&corpus_path)
        .map_err(|err| format!("failed to read {corpus_path}: {err}"))?;
    let corpus: Corpus =
        serde_json::from_str(&corpus_text).map_err(|err| format!("bad corpus: {err}"))?;
    let meetings: Vec<CorpusMeeting> = corpus
        .meetings
        .into_iter()
        .filter(|m| {
            meeting_filter
                .as_ref()
                .is_none_or(|set| set.contains(&m.id))
        })
        .collect();
    if meetings.is_empty() {
        return Err("corpus has no selected meetings".to_string());
    }
    let candidates: Vec<(usize, Candidate)> = CANDIDATES
        .iter()
        .enumerate()
        .filter(|(_, c)| model_filter.as_ref().is_none_or(|set| set.contains(c.id)))
        .map(|(index, candidate)| (index, *candidate))
        .collect();
    if candidates.is_empty() {
        return Err("no selected candidates".to_string());
    }
    let judges: Vec<Judge> = JUDGES
        .iter()
        .filter(|j| judge_filter.as_ref().is_none_or(|set| set.contains(j.id())))
        .copied()
        .collect();
    if judges.is_empty() {
        return Err("no selected judges".to_string());
    }

    let clients = Arc::new(build_clients(&keys_dir)?);
    let store = Store::open(&run_dir)?;
    let effort = effort_from_env();
    let blind_labels: HashMap<String, String> = CANDIDATES
        .iter()
        .enumerate()
        .map(|(index, c)| (c.id.to_string(), blind_label(index)))
        .collect();
    let failures = Arc::new(AtomicUsize::new(0));
    // One meta row per invocation: reruns with different filters stay honest.
    store.append(&MetaRecord {
        kind: "meta".to_string(),
        started_at_ms: now_ms(),
        candidates: candidates.iter().map(|(_, c)| c.id.to_string()).collect(),
        judges: judges.iter().map(|j| j.id().to_string()).collect(),
        meetings: meetings.iter().map(|m| m.id.clone()).collect(),
        blind_labels: blind_labels.clone(),
        effort: effort.clone().unwrap_or_else(|| "default".to_string()),
    })?;

    // Phase 1: generate.
    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let (gen_tx, mut gen_rx) = tokio::sync::mpsc::unbounded_channel::<GenRecord>();
    let mut pending = 0usize;
    let mut skipped = 0usize;
    for meeting in &meetings {
        let (system, user) = render_prompts(meeting)?;
        let prompt_hash = fnv1a_hex(&format!("{system}\n{user}"));
        for (index, candidate) in &candidates {
            if store
                .done_gens
                .contains(&(meeting.id.clone(), candidate.id.to_string()))
            {
                skipped += 1;
                continue;
            }
            pending += 1;
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|err| err.to_string())?;
            let tx = gen_tx.clone();
            let failures = failures.clone();
            let clients = clients.clone();
            let meeting_id = meeting.id.clone();
            let candidate = *candidate;
            let blind = blind_label(*index);
            let (system, user) = (system.clone(), user.clone());
            let prompt_hash = prompt_hash.clone();
            let effort = effort.clone();
            tokio::spawn(async move {
                let _permit = permit;
                let started = Instant::now();
                let result =
                    generate_note(&clients, candidate, &system, &user, effort.as_deref()).await;
                match result {
                    Ok((output, input_tokens, output_tokens, cost_usd)) => {
                        let _ = tx.send(GenRecord {
                            kind: "gen".to_string(),
                            meeting: meeting_id,
                            model: candidate.id.to_string(),
                            blind,
                            output,
                            input_tokens,
                            output_tokens,
                            cost_usd,
                            elapsed_ms: started.elapsed().as_millis(),
                            prompt_hash,
                        });
                    }
                    Err(err) => {
                        failures.fetch_add(1, Ordering::Relaxed);
                        eprintln!("gen failed {meeting_id} {}: {err}", candidate.id);
                    }
                }
            });
        }
    }
    drop(gen_tx);
    let mut generated = 0usize;
    while let Some(record) = gen_rx.recv().await {
        store.append(&record)?;
        generated += 1;
        println!(
            "gen {generated}/{pending} {} {}",
            record.meeting, record.model
        );
    }
    println!("phase 1 done: {generated} new, {skipped} resumed");

    // Phase 2: judge every recorded generation.
    let run_text = std::fs::read_to_string(&store.path).unwrap_or_default();
    let mut outputs: HashMap<(String, String), (String, String)> = HashMap::new();
    for line in run_text.lines() {
        if let Ok(record) = serde_json::from_str::<GenRecord>(line) {
            outputs.insert(
                (record.meeting, record.model),
                (record.blind, record.output),
            );
        }
    }
    let (judge_tx, mut judge_rx) = tokio::sync::mpsc::unbounded_channel::<JudgeRecord>();
    let mut judge_pending = 0usize;
    let mut judge_skipped = 0usize;
    for meeting in &meetings {
        let full_transcript = transcript_text(meeting);
        for (_, candidate) in &candidates {
            let key = (meeting.id.clone(), candidate.id.to_string());
            let Some((blind, output)) = outputs.get(&key) else {
                failures.fetch_add(1, Ordering::Relaxed);
                eprintln!("judge skipped, no output: {} {}", meeting.id, candidate.id);
                continue;
            };
            let fitted_note = fit_note(output);
            let (transcript, truncated, transcript_chars) =
                fit_judge_input(&full_transcript, &[&fitted_note]);
            for judge in &judges {
                if store.done_judges.contains(&(
                    meeting.id.clone(),
                    candidate.id.to_string(),
                    judge.id().to_string(),
                )) {
                    judge_skipped += 1;
                    continue;
                }
                judge_pending += 1;
                let permit = semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|err| err.to_string())?;
                let tx = judge_tx.clone();
                let failures = failures.clone();
                let clients = clients.clone();
                let (meeting_id, model, blind, output, transcript) = (
                    meeting.id.clone(),
                    candidate.id.to_string(),
                    blind.clone(),
                    fitted_note.clone(),
                    transcript.clone(),
                );
                let judge = *judge;
                tokio::spawn(async move {
                    let _permit = permit;
                    let started = Instant::now();
                    match judge_note(&clients, judge, &blind, &transcript, &output).await {
                        Ok((scores, input_tokens, output_tokens, cost_usd)) => {
                            let _ = tx.send(JudgeRecord {
                                kind: "judge".to_string(),
                                meeting: meeting_id,
                                model,
                                judge: judge.id().to_string(),
                                scores,
                                evidence: HashMap::new(),
                                input_tokens,
                                output_tokens,
                                cost_usd,
                                elapsed_ms: started.elapsed().as_millis(),
                                truncated,
                                transcript_chars,
                            });
                        }
                        Err(err) => {
                            failures.fetch_add(1, Ordering::Relaxed);
                            eprintln!("judge failed {meeting_id} {model} {}: {err}", judge.id());
                        }
                    }
                });
            }
        }
    }
    drop(judge_tx);
    let mut judged = 0usize;
    while let Some(record) = judge_rx.recv().await {
        store.append(&record)?;
        judged += 1;
        println!(
            "judge {judged}/{judge_pending} {} {} {}",
            record.meeting, record.model, record.judge
        );
    }
    println!("phase 2 done: {judged} new, {judge_skipped} resumed");

    // Phase 3: pairwise ranking per judge.
    let (pair_tx, mut pair_rx) = tokio::sync::mpsc::unbounded_channel::<PairRecord>();
    let mut pair_pending = 0usize;
    let mut pair_skipped = 0usize;
    for meeting in &meetings {
        let full_transcript = transcript_text(meeting);
        for judge in &judges {
            for (left_pos, (_, left)) in candidates.iter().enumerate() {
                for (_, right) in candidates.iter().skip(left_pos + 1) {
                    let (model_a, model_b) = if left.id <= right.id {
                        (left.id.to_string(), right.id.to_string())
                    } else {
                        (right.id.to_string(), left.id.to_string())
                    };
                    if store.done_pairs.contains(&(
                        meeting.id.clone(),
                        judge.id().to_string(),
                        model_a.clone(),
                        model_b.clone(),
                    )) {
                        pair_skipped += 1;
                        continue;
                    }
                    let left_out = outputs.get(&(meeting.id.clone(), left.id.to_string()));
                    let right_out = outputs.get(&(meeting.id.clone(), right.id.to_string()));
                    let (Some((blind_l, out_l)), Some((blind_r, out_r))) = (left_out, right_out)
                    else {
                        failures.fetch_add(1, Ordering::Relaxed);
                        eprintln!(
                            "pair skipped, no output: {} {} vs {}",
                            meeting.id, left.id, right.id
                        );
                        continue;
                    };
                    let swap = pair_swapped(&meeting.id, judge.id(), &model_a, &model_b);
                    let (blind_first, note_first, first, blind_second, note_second) = if swap {
                        (
                            blind_r.clone(),
                            fit_note(out_r),
                            right.id.to_string(),
                            blind_l.clone(),
                            fit_note(out_l),
                        )
                    } else {
                        (
                            blind_l.clone(),
                            fit_note(out_l),
                            left.id.to_string(),
                            blind_r.clone(),
                            fit_note(out_r),
                        )
                    };
                    let (transcript, _, _) =
                        fit_judge_input(&full_transcript, &[&note_first, &note_second]);
                    pair_pending += 1;
                    let permit = semaphore
                        .clone()
                        .acquire_owned()
                        .await
                        .map_err(|err| err.to_string())?;
                    let tx = pair_tx.clone();
                    let failures = failures.clone();
                    let clients = clients.clone();
                    let (meeting_id, transcript) = (meeting.id.clone(), transcript.clone());
                    let judge = *judge;
                    tokio::spawn(async move {
                        let _permit = permit;
                        let started = Instant::now();
                        let state = pair_state(
                            &blind_first,
                            &note_first,
                            &blind_second,
                            &note_second,
                            &transcript,
                        );
                        let result = clients
                            .judge_client(judge)
                            .judge_pair(judge.model(), &state)
                            .await;
                        match result {
                            Ok(response) => match anlg_template_eval::choice_p_a(&response) {
                                Some(p_first) => {
                                    let (input_tokens, output_tokens) = response
                                        .usage
                                        .as_ref()
                                        .map(|usage| (usage.input_tokens, usage.output_tokens))
                                        .unwrap_or((0, 0));
                                    let cost_usd = None;
                                    let _ = tx.send(PairRecord {
                                        kind: "pair".to_string(),
                                        meeting: meeting_id,
                                        judge: judge.id().to_string(),
                                        model_a,
                                        model_b,
                                        first,
                                        p_first,
                                        input_tokens,
                                        output_tokens,
                                        cost_usd,
                                        elapsed_ms: started.elapsed().as_millis(),
                                    });
                                }
                                None => {
                                    failures.fetch_add(1, Ordering::Relaxed);
                                    eprintln!(
                                        "pair failed {meeting_id} {}: no choice answer",
                                        judge.id()
                                    );
                                }
                            },
                            Err(err) => {
                                failures.fetch_add(1, Ordering::Relaxed);
                                eprintln!("pair failed {meeting_id} {}: {err}", judge.id());
                            }
                        }
                    });
                }
            }
        }
    }
    drop(pair_tx);
    let mut paired = 0usize;
    while let Some(record) = pair_rx.recv().await {
        store.append(&record)?;
        paired += 1;
        println!(
            "pair {paired}/{pair_pending} {} {} {} vs {}",
            record.meeting, record.judge, record.model_a, record.model_b
        );
    }
    println!("phase 3 done: {paired} new, {pair_skipped} resumed");
    let failed = failures.load(Ordering::Relaxed);
    if failed > 0 {
        return Err(format!(
            "{failed} generation/judgment calls failed; fix and rerun (finished items resume)"
        ));
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("run");
    let result = match command {
        "run" => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");
            runtime.block_on(run())
        }
        "report" => report::render(),
        _ => Err(format!("unknown command {command}; want run|report")),
    };
    if let Err(err) = result {
        eprintln!("meeting_bench {command} failed: {err}");
        std::process::exit(1);
    }
}
