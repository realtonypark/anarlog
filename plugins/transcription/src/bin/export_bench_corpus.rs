//! Export meeting transcripts for the meeting-notes benchmark corpus.
//!
//! Uses the production renderer (`render_session_transcript`), so bench inputs
//! match what the app feeds the enhance prompt. Read-only: opens the database
//! with `mode=ro` and never writes to it.
//!
//! Usage: export_bench_corpus <app.db> <corpus.json> [max_meetings]
//! The corpus holds private meeting content: keep it outside the repository.

use sqlx::Row;
use sqlx::sqlite::SqliteConnectOptions;
use tauri_plugin_transcription::session_transcript::{
    RenderSessionTranscriptRequest, render_session_transcript,
};

#[derive(Debug, serde::Serialize)]
struct CorpusSegment {
    speaker: String,
    text: String,
    start_ms: Option<i64>,
}

#[derive(Debug, serde::Serialize)]
struct CorpusParticipant {
    name: String,
    job_title: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct CorpusMeeting {
    id: String,
    title: String,
    started_at: String,
    ended_at: String,
    language: String,
    participants: Vec<CorpusParticipant>,
    segments: Vec<CorpusSegment>,
    transcript_chars: usize,
}

#[derive(Debug, serde::Serialize)]
struct Corpus {
    exported_at_ms: u128,
    source_db: String,
    renderer: String,
    meetings: Vec<CorpusMeeting>,
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

async fn run(db_path: &str, max_meetings: usize) -> Result<Corpus, String> {
    let options = SqliteConnectOptions::new()
        .filename(db_path)
        .read_only(true)
        .create_if_missing(false);
    let pool = sqlx::SqlitePool::connect_with(options)
        .await
        .map_err(|err| format!("failed to open db read-only: {err}"))?;

    let sessions = sqlx::query(
        "SELECT id, title, started_at, ended_at, language, owner_user_id
         FROM sessions
         WHERE deleted_at IS NULL
           AND id IN (SELECT session_id FROM transcripts)
         ORDER BY created_at DESC",
    )
    .fetch_all(&pool)
    .await
    .map_err(|err| format!("failed to list sessions: {err}"))?;

    let mut meetings = Vec::new();
    for row in sessions {
        if meetings.len() >= max_meetings {
            break;
        }
        let id: String = row.get("id");
        let owner: String = row.get("owner_user_id");
        let rendered = render_session_transcript(
            &pool,
            RenderSessionTranscriptRequest {
                session_id: id.clone(),
                self_human_id: (!owner.is_empty()).then(|| owner.clone()),
            },
        )
        .await
        .map_err(|err| format!("failed to render {id}: {err}"))?;
        let Some(rendered) = rendered else {
            continue;
        };
        let segments: Vec<CorpusSegment> = rendered
            .segments
            .iter()
            .filter(|segment| !segment.text.trim().is_empty())
            .map(|segment| CorpusSegment {
                speaker: segment.speaker_label.clone(),
                text: segment.text.clone(),
                start_ms: Some(segment.start_ms),
            })
            .collect();
        if segments.is_empty() {
            continue;
        }
        let participant_rows = sqlx::query(
            "SELECT COALESCE(NULLIF(h.name, ''), p.display_name) AS name, h.job_title
             FROM session_participants AS p
             LEFT JOIN humans AS h ON h.id = p.human_id AND h.deleted_at IS NULL
             WHERE p.session_id = ? AND p.deleted_at IS NULL AND p.source <> 'excluded'
             ORDER BY name",
        )
        .bind(&id)
        .fetch_all(&pool)
        .await
        .map_err(|err| format!("failed to load participants for {id}: {err}"))?;
        let participants = participant_rows
            .iter()
            .filter_map(|row| {
                let name: Option<String> = row.get("name");
                name.filter(|name| !name.trim().is_empty())
                    .map(|name| CorpusParticipant {
                        name,
                        job_title: row.get("job_title"),
                    })
            })
            .collect::<Vec<_>>();
        let transcript_chars = segments.iter().map(|s| s.text.chars().count()).sum();
        let title: String = row.get("title");
        meetings.push(CorpusMeeting {
            id,
            title,
            started_at: row.get("started_at"),
            ended_at: row.get("ended_at"),
            language: row.get("language"),
            participants,
            segments,
            transcript_chars,
        });
    }

    Ok(Corpus {
        exported_at_ms: now_ms(),
        source_db: std::path::Path::new(db_path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("app.db")
            .to_string(),
        renderer: "render_session_transcript".to_string(),
        meetings,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: export_bench_corpus <app.db> <corpus.json> [max_meetings]");
        std::process::exit(2);
    }
    let max_meetings = args
        .get(3)
        .and_then(|value| value.parse().ok())
        .unwrap_or(usize::MAX);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let corpus = runtime
        .block_on(run(&args[1], max_meetings))
        .unwrap_or_else(|err| {
            eprintln!("export failed: {err}");
            std::process::exit(1);
        });
    let json = serde_json::to_string_pretty(&corpus).expect("serialize corpus");
    let tmp = format!("{}.tmp", &args[2]);
    std::fs::write(&tmp, json).unwrap_or_else(|err| {
        eprintln!("write failed: {err}");
        std::process::exit(1);
    });
    std::fs::rename(&tmp, &args[2]).unwrap_or_else(|err| {
        eprintln!("rename failed: {err}");
        std::process::exit(1);
    });
    let total_chars: usize = corpus.meetings.iter().map(|m| m.transcript_chars).sum();
    println!(
        "exported {} meetings ({} transcript chars) -> {}",
        corpus.meetings.len(),
        total_chars,
        args[2]
    );
}
