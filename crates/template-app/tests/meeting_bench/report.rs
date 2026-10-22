//! report: run.jsonl -> report.md + index.html (Vals-style, results only).
//!
//! No transcript or note text leaves the run dir: both outputs carry scores,
//! costs, and method summaries only. Meetings appear as "Meeting N" labels.

use std::collections::{HashMap, HashSet};

use anlg_template_eval::{bootstrap_ci, mean, mean_spread, pearson};

use crate::CANDIDATES;

const CRITERIA: [(&str, &str); 6] = [
    ("grounded", "Faithfulness"),
    ("decisions", "Decisions"),
    ("actions", "Actions"),
    ("coverage", "Coverage"),
    ("attribution", "Attribution"),
    ("concise", "Concision"),
];

const JUDGE_NAMES: [(&str, &str); 3] = [
    ("mammoth", "Mammoth 27B (Baseten decision model)"),
    ("jev", "Jev 1.14 (Ramp Router decision model)"),
    (
        "decider",
        "Decider V1.2 (Perplexity decision model via OpenRouter)",
    ),
];

/// Verified 2026-10-21: (model id, $/1M input, $/1M output). Ramp prices from
/// router.com/models/<slug> (catalog Oct 15 2026); OpenRouter from /api/v1/models.
/// Mammoth has no published per-token price (dashboard total only).
const PRICES: [(&str, f64, f64); 10] = [
    ("claude-haiku-5-5", 0.10, 0.50),
    ("gpt-6-luna", 0.10, 0.50),
    ("deepseek-v4.1-flash-1017", 0.30, 1.20),
    ("glm-5p3-flash", 0.15, 0.50),
    ("grok-4.6", 2.00, 6.00),
    ("google/gemini-3.8-flash", 0.825, 4.125),
    ("nvidia/nemotron-3.5-super-vl", 0.15, 0.50),
    ("meta/muse-spark-1.3-contributor", 0.10, 0.20),
    ("jev-1.14.0", 0.042, 0.00),
    ("perplexity/pplx-decider-v1.2", 0.02, 0.00),
];

fn estimate_usd(model: &str, input_tokens: u64, output_tokens: u64) -> Option<f64> {
    PRICES
        .iter()
        .find(|(id, _, _)| *id == model)
        .map(|(_, input, output)| {
            input_tokens as f64 / 1_000_000.0 * input + output_tokens as f64 / 1_000_000.0 * output
        })
}

fn judge_model_id(judge: &str) -> Option<&'static str> {
    match judge {
        "mammoth" => None,
        "jev" => Some("jev-1.14.0"),
        "decider" => Some("perplexity/pplx-decider-v1.2"),
        _ => None,
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct GenRow {
    #[serde(default)]
    meeting: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct JudgeRow {
    #[serde(default)]
    meeting: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    judge: String,
    #[serde(default)]
    scores: HashMap<String, f64>,
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    truncated: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Probe {
    #[serde(rename = "t")]
    kind: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct PairRow {
    #[serde(default)]
    meeting: String,
    #[serde(default)]
    judge: String,
    #[serde(default)]
    model_a: String,
    #[serde(default)]
    model_b: String,
    #[serde(default)]
    first: String,
    #[serde(default)]
    p_first: f64,
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
struct MetaRow {
    #[serde(default)]
    candidates: Vec<String>,
    #[serde(default)]
    judges: Vec<String>,
    #[serde(default)]
    meetings: Vec<String>,
    #[serde(default)]
    effort: String,
}

struct ModelRow {
    id: String,
    overall: f64,
    ci: (f64, f64),
    criteria: HashMap<String, f64>,
    borda: u32,
    pair_pooled: f64,
    judge_ranks: HashMap<String, usize>,
    gen_in: u64,
    gen_out: u64,
}

struct Report {
    models: Vec<ModelRow>,
    meeting_ids: Vec<String>,
    meeting_label: HashMap<String, String>,
    item_overall: HashMap<(String, String), f64>,
    judge_pairs: Vec<(String, String, Option<f64>)>,
    spread: Option<f64>,
    judge_tokens: HashMap<String, (u64, u64)>,
    judge_rankings: Vec<(String, Vec<(String, u32, f64)>)>,
    meeting_winner: HashMap<String, String>,
    judgments: usize,
    generations: usize,
    pairs: usize,
    truncated_notes: usize,
    effort: String,
}

fn load(run_dir: &str) -> Result<Report, String> {
    let text = std::fs::read_to_string(format!("{run_dir}/run.jsonl"))
        .map_err(|err| format!("failed to read run.jsonl: {err}"))?;
    let mut gens: Vec<GenRow> = Vec::new();
    let mut judges: Vec<JudgeRow> = Vec::new();
    let mut pairs: Vec<PairRow> = Vec::new();
    let mut meta = MetaRow::default();
    for line in text.lines() {
        let Ok(probe) = serde_json::from_str::<Probe>(line) else {
            continue;
        };
        match probe.kind.as_str() {
            "gen" => {
                if let Ok(row) = serde_json::from_str::<GenRow>(line) {
                    gens.push(row);
                }
            }
            "judge" => {
                if let Ok(row) = serde_json::from_str::<JudgeRow>(line) {
                    judges.push(row);
                }
            }
            "meta" => {
                if let Ok(row) = serde_json::from_str::<MetaRow>(line) {
                    meta = row;
                }
            }
            "pair" => {
                if let Ok(row) = serde_json::from_str::<PairRow>(line) {
                    pairs.push(row);
                }
            }
            _ => {}
        }
    }
    // Dedupe by key keeping last: filter-change reruns must not double-count.
    let mut gen_seen = HashSet::new();
    gens = gens
        .into_iter()
        .rev()
        .filter(|row| gen_seen.insert((row.meeting.clone(), row.model.clone())))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let mut judge_seen = HashSet::new();
    judges = judges
        .into_iter()
        .rev()
        .filter(|row| {
            judge_seen.insert((row.meeting.clone(), row.model.clone(), row.judge.clone()))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let mut pair_seen = HashSet::new();
    pairs = pairs
        .into_iter()
        .rev()
        .filter(|row| {
            let (mut a, mut b) = (row.model_a.clone(), row.model_b.clone());
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            pair_seen.insert((row.meeting.clone(), row.judge.clone(), a, b))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if gens.is_empty() || judges.is_empty() {
        return Err("run.jsonl has no generations or judgments yet".to_string());
    }
    // Pair spend folds into judge totals.
    let mut judge_tokens: HashMap<String, (u64, u64)> = HashMap::new();
    for row in &pairs {
        let entry = judge_tokens.entry(row.judge.clone()).or_insert((0, 0));
        entry.0 += row.input_tokens;
        entry.1 += row.output_tokens;
    }

    let mut meeting_ids: Vec<String> = gens.iter().map(|g| g.meeting.clone()).collect();
    meeting_ids.sort();
    meeting_ids.dedup();
    let meeting_label: HashMap<String, String> = meeting_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), format!("Meeting {}", i + 1)))
        .collect();

    // Per (meeting, model, judge): mean over the 6 required criteria.
    let mut judge_overall: HashMap<(String, String, String), f64> = HashMap::new();
    let mut truncated_notes = 0usize;
    for row in &judges {
        let values: Vec<f64> = CRITERIA
            .iter()
            .filter_map(|(id, _)| row.scores.get(*id).copied())
            .collect();
        if values.len() == CRITERIA.len()
            && let Some(score) = mean(&values)
        {
            judge_overall.insert(
                (row.meeting.clone(), row.model.clone(), row.judge.clone()),
                score,
            );
        }
        if row.truncated {
            truncated_notes += 1;
        }
        let entry = judge_tokens.entry(row.judge.clone()).or_insert((0, 0));
        entry.0 += row.input_tokens;
        entry.1 += row.output_tokens;
    }

    // Per (meeting, model): per-criterion mean over judges first, so columns
    // decompose the overall; items need >= 2 judges.
    let mut item_criteria: HashMap<(String, String), HashMap<String, f64>> = HashMap::new();
    let mut item_judges: HashMap<(String, String), HashMap<String, f64>> = HashMap::new();
    let mut by_item: HashMap<(String, String), Vec<&JudgeRow>> = HashMap::new();
    for row in &judges {
        by_item
            .entry((row.meeting.clone(), row.model.clone()))
            .or_default()
            .push(row);
    }
    for (key, rows) in &by_item {
        if rows.len() < 2 {
            continue;
        }
        let mut criteria = HashMap::new();
        for (criterion, _) in CRITERIA {
            let values: Vec<f64> = rows
                .iter()
                .filter_map(|row| row.scores.get(criterion).copied())
                .collect();
            if let Some(score) = mean(&values) {
                criteria.insert(criterion.to_string(), score);
            }
        }
        if criteria.len() == CRITERIA.len() {
            item_criteria.insert(key.clone(), criteria);
        }
        let mut per_judge = HashMap::new();
        for row in rows {
            if let Some(score) =
                judge_overall.get(&(row.meeting.clone(), row.model.clone(), row.judge.clone()))
            {
                per_judge.insert(row.judge.clone(), *score);
            }
        }
        if per_judge.len() >= 2 {
            item_judges.insert(key.clone(), per_judge);
        }
    }
    let mut item_overall: HashMap<(String, String), f64> = HashMap::new();
    for (key, criteria) in &item_criteria {
        if let Some(overall) = mean(&criteria.values().copied().collect::<Vec<_>>()) {
            item_overall.insert(key.clone(), overall);
        }
    }

    // Completeness against the last meta row: partial runs fail loudly.
    if !meta.candidates.is_empty() && !meta.meetings.is_empty() {
        let mut gaps = Vec::new();
        for meeting in &meta.meetings {
            for model in &meta.candidates {
                let judges_present = item_judges
                    .get(&(meeting.clone(), model.clone()))
                    .map(|scores| scores.len())
                    .unwrap_or(0);
                if judges_present < 2 {
                    gaps.push(format!(
                        "{meeting} {model}: {judges_present}/{} judges",
                        meta.judges.len().max(1)
                    ));
                }
            }
        }
        if !gaps.is_empty() {
            let shown = gaps.iter().take(10).cloned().collect::<Vec<_>>().join("; ");
            return Err(format!(
                "incomplete coverage ({} gaps, e.g. {shown}); rerun to resume",
                gaps.len()
            ));
        }
    }

    // Judge agreement on shared items.
    let judge_ids: Vec<String> = JUDGE_NAMES.iter().map(|(id, _)| id.to_string()).collect();
    let mut judge_pairs = Vec::new();
    for (i, left) in judge_ids.iter().enumerate() {
        for right in judge_ids.iter().skip(i + 1) {
            let mut l = Vec::new();
            let mut r = Vec::new();
            for scores in item_judges.values() {
                if let (Some(a), Some(b)) = (scores.get(left), scores.get(right)) {
                    l.push(*a);
                    r.push(*b);
                }
            }
            judge_pairs.push((left.clone(), right.clone(), pearson(&l, &r)));
        }
    }
    let spread = mean_spread(
        &item_judges
            .values()
            .map(|scores| scores.values().copied().collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    );

    // Axis 2: per (meeting, judge) rank by mean P(chosen), then Borda.
    // Each meeting is one contest (equal weight); each judge one ballot.
    if pairs.is_empty() {
        return Err("no pairwise judgments; run phase 3 first".to_string());
    }
    if !meta.candidates.is_empty() && !meta.meetings.is_empty() && !meta.judges.is_empty() {
        let expected = meta.candidates.len() * meta.candidates.len().saturating_sub(1) / 2;
        let mut have: HashMap<(String, String), usize> = HashMap::new();
        for row in &pairs {
            *have
                .entry((row.meeting.clone(), row.judge.clone()))
                .or_default() += 1;
        }
        let gaps = meta
            .meetings
            .iter()
            .flat_map(|meeting| meta.judges.iter().map(move |judge| (meeting, judge)))
            .filter(|(meeting, judge)| {
                have.get(&(meeting.to_string(), judge.to_string()))
                    .copied()
                    .unwrap_or(0)
                    < expected
            })
            .count();
        if gaps > 0 {
            return Err(format!(
                "incomplete pair coverage ({gaps} meeting-judge cells short of {expected} pairs); rerun to resume"
            ));
        }
    }
    let mut pair_p: HashMap<(String, String, String), Vec<f64>> = HashMap::new();
    for row in &pairs {
        if !(0.0..=1.0).contains(&row.p_first) {
            continue;
        }
        let p_a = if row.first == row.model_a {
            row.p_first
        } else {
            1.0 - row.p_first
        };
        pair_p
            .entry((row.meeting.clone(), row.judge.clone(), row.model_a.clone()))
            .or_default()
            .push(p_a);
        pair_p
            .entry((row.meeting.clone(), row.judge.clone(), row.model_b.clone()))
            .or_default()
            .push(1.0 - p_a);
    }
    let mut by_meeting_judge: HashMap<(String, String), Vec<(String, f64)>> = HashMap::new();
    for ((meeting, judge, model), values) in &pair_p {
        if let Some(score) = mean(values) {
            by_meeting_judge
                .entry((meeting.clone(), judge.clone()))
                .or_default()
                .push((model.clone(), score));
        }
    }
    let mut borda: HashMap<(String, String), u32> = HashMap::new();
    let mut meeting_points: HashMap<(String, String), u32> = HashMap::new();
    for ((meeting, judge), entrants) in &by_meeting_judge {
        let mut ranked = entrants.clone();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let n = ranked.len() as u32;
        for (pos, (model, _)) in ranked.iter().enumerate() {
            let points = n - pos as u32;
            *borda.entry((judge.clone(), model.clone())).or_default() += points;
            *meeting_points
                .entry((meeting.clone(), model.clone()))
                .or_default() += points;
        }
    }
    let mut pooled_values: HashMap<(String, String), Vec<f64>> = HashMap::new();
    for ((_, judge, model), values) in &pair_p {
        pooled_values
            .entry((judge.clone(), model.clone()))
            .or_default()
            .extend(values.iter().copied());
    }
    let pooled_p: HashMap<(String, String), f64> = pooled_values
        .into_iter()
        .filter_map(|(key, values)| mean(&values).map(|m| (key, m)))
        .collect();
    let mut judge_rankings: Vec<(String, Vec<(String, u32, f64)>)> = Vec::new();
    for (judge_id, _) in JUDGE_NAMES {
        let mut list: Vec<(String, u32, f64)> = borda
            .iter()
            .filter(|((judge, _), _)| judge == judge_id)
            .map(|((_, model), points)| {
                let pooled = pooled_p
                    .get(&(judge_id.to_string(), model.clone()))
                    .copied()
                    .unwrap_or(0.0);
                (model.clone(), *points, pooled)
            })
            .collect();
        list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        judge_rankings.push((judge_id.to_string(), list));
    }
    let judge_rank_of = |judge: &str, model: &str| -> Option<usize> {
        judge_rankings
            .iter()
            .find(|(id, _)| id == judge)
            .and_then(|(_, list)| list.iter().position(|(m, _, _)| m == model))
            .map(|pos| pos + 1)
    };
    let mut meeting_winner: HashMap<String, String> = HashMap::new();
    for id in meeting_ids.iter() {
        let mut entrants: Vec<(String, u32)> = meeting_points
            .iter()
            .filter(|((meeting, _), _)| meeting == id)
            .map(|((_, model), points)| (model.clone(), *points))
            .collect();
        entrants.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if let Some((winner, _)) = entrants.first() {
            meeting_winner.insert(id.clone(), winner.clone());
        }
    }

    // Per-model rows, sorted by Borda total, then absolute overall.
    let mut models = Vec::new();
    for candidate in CANDIDATES {
        let id = candidate.id.to_string();
        let mut per_criterion: HashMap<String, Vec<f64>> = HashMap::new();
        let mut overalls = Vec::new();
        let mut keys: Vec<&(String, String)> = item_criteria
            .keys()
            .filter(|(_, model)| *model == id)
            .collect();
        keys.sort();
        for key in keys {
            let criteria = &item_criteria[key];
            for (criterion, _) in CRITERIA {
                if let Some(score) = criteria.get(criterion) {
                    per_criterion
                        .entry(criterion.to_string())
                        .or_default()
                        .push(*score);
                }
            }
            if let Some(score) = item_overall.get(key) {
                overalls.push(*score);
            }
        }
        overalls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let Some(overall) = mean(&overalls) else {
            continue;
        };
        let ci = bootstrap_ci(&overalls, 1000).unwrap_or((overall, overall));
        let criteria = per_criterion
            .into_iter()
            .filter_map(|(key, values)| mean(&values).map(|m| (key, m)))
            .collect();
        let (gen_in, gen_out) = gens
            .iter()
            .filter(|g| g.model == id)
            .fold((0, 0), |(i, o), g| {
                (i + g.input_tokens, o + g.output_tokens)
            });
        let borda_total: u32 = JUDGE_NAMES
            .iter()
            .filter_map(|(judge, _)| borda.get(&(judge.to_string(), id.clone())))
            .sum();
        let pair_means: Vec<f64> = JUDGE_NAMES
            .iter()
            .filter_map(|(judge, _)| pooled_p.get(&(judge.to_string(), id.clone())).copied())
            .collect();
        let judge_ranks: HashMap<String, usize> = JUDGE_NAMES
            .iter()
            .filter_map(|(judge, _)| {
                judge_rank_of(judge, &id).map(|rank| (judge.to_string(), rank))
            })
            .collect();
        models.push(ModelRow {
            id,
            overall,
            ci,
            criteria,
            borda: borda_total,
            pair_pooled: mean(&pair_means).unwrap_or(0.0),
            judge_ranks,
            gen_in,
            gen_out,
        });
    }
    models.sort_by(|a, b| {
        b.borda
            .cmp(&a.borda)
            .then_with(|| {
                b.overall
                    .partial_cmp(&a.overall)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.id.cmp(&b.id))
    });

    Ok(Report {
        models,
        meeting_ids,
        meeting_label,
        item_overall,
        judge_pairs,
        spread,
        judge_tokens,
        judge_rankings,
        meeting_winner,
        judgments: judges.len(),
        generations: gens.len(),
        pairs: pairs.len(),
        truncated_notes,
        effort: if meta.effort.is_empty() {
            "default".to_string()
        } else {
            meta.effort
        },
    })
}

fn pct(score: f64) -> String {
    format!("{:.1}", score * 100.0)
}

fn esc(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn usd(value: f64) -> String {
    format!("${value:.2}")
}

fn judge_cost_text(report: &Report, judge: &str, dashboard_label: &str) -> String {
    let (input, output) = report.judge_tokens.get(judge).copied().unwrap_or((0, 0));
    judge_model_id(judge)
        .and_then(|model| estimate_usd(model, input, output))
        .map(|est| format!("~{} (computed)", usd(est)))
        .unwrap_or_else(|| format!("n/a ({dashboard_label})"))
}

fn env_usd(key: &str) -> Option<f64> {
    std::env::var(key).ok()?.parse().ok()
}

fn render_markdown(report: &Report) -> String {
    let mut out = String::from("# Meeting-note summarization benchmark\n\n");
    out.push_str("Which LLM writes the best meeting note from a transcript, using the production Analog enhance prompt. Eight candidates, one note per meeting each, judged blind by three decision models.\n\n");
    out.push_str("Two axes. (1) Absolute: each judge scores every note on six criteria; the diagnostic tables show means. (2) Ranking: each judge picks the better note for every pair within each meeting; models rank per meeting per judge, and the final leaderboard is Borda over all meetings and judges (equal weight per meeting per judge).\n\n");
    out.push_str("Each candidate gets the transcript plus session title and participants through the production enhance template (default format, no output template, no pre/post memos, no calendar event).\n\n");
    out.push_str(&format!(
        "Meetings: {} | Candidates: {} | Absolute judgments: {} | Pairwise: {} | Generations: {}\n\n",
        report.meeting_ids.len(),
        report.models.len(),
        report.judgments,
        report.pairs,
        report.generations
    ));
    out.push_str("## Leaderboard (Borda over per-meeting pairwise ranks)\n\n");
    out.push_str("| Rank | Model | Borda | Mammoth | Jev | Decider | Pair P | Absolute |\n");
    out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for (rank, row) in report.models.iter().enumerate() {
        let rank_cell = |judge: &str| {
            row.judge_ranks
                .get(judge)
                .map(|rank| rank.to_string())
                .unwrap_or_else(|| "-".to_string())
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            rank + 1,
            row.id,
            row.borda,
            rank_cell("mammoth"),
            rank_cell("jev"),
            rank_cell("decider"),
            pct(row.pair_pooled),
            pct(row.overall),
        ));
    }
    out.push_str("\n## Absolute scores by criterion (diagnostic)\n\n");
    out.push_str("| Model | Overall | 95% CI |");
    for (_, label) in CRITERIA {
        out.push_str(&format!(" {label} |"));
    }
    out.push_str("\n| --- | --- | --- |");
    for _ in CRITERIA {
        out.push_str(" --- |");
    }
    out.push('\n');
    for row in &report.models {
        out.push_str(&format!(
            "| {} | {} | [{}, {}] |",
            row.id,
            pct(row.overall),
            pct(row.ci.0),
            pct(row.ci.1)
        ));
        for (id, _) in CRITERIA {
            let cell = row
                .criteria
                .get(id)
                .map(|score| pct(*score))
                .unwrap_or_else(|| "-".to_string());
            out.push_str(&format!(" {cell} |"));
        }
        out.push('\n');
    }
    out.push_str("\n## Judges\n\n");
    for (id, name) in JUDGE_NAMES {
        out.push_str(&format!("- {name} (`{id}`)\n"));
    }
    out.push_str("\nEach judge answers one yes/no question per criterion and returns P(yes), and picks the better note for every pair. Model names are hidden behind blind labels during grading.\n\n");
    out.push_str("## Per-judge rankings (Borda over meetings)\n\n");
    for (judge, list) in &report.judge_rankings {
        out.push_str(&format!("### {judge}\n\n"));
        for (pos, (model, points, pooled)) in list.iter().enumerate() {
            out.push_str(&format!(
                "{}. {} ({} pts, P={})\n",
                pos + 1,
                model,
                points,
                pct(*pooled)
            ));
        }
        out.push('\n');
    }
    out.push_str("## Judge agreement\n\n");
    for (left, right, rho) in &report.judge_pairs {
        out.push_str(&format!(
            "- {left} vs {right}: Pearson {}\n",
            rho.map(|r| format!("{r:.2}"))
                .unwrap_or_else(|| "n/a".to_string())
        ));
    }
    out.push_str(&format!(
        "- Mean judge spread per note: {}\n",
        report
            .spread
            .map(|s| format!("{:.1} pts", s * 100.0))
            .unwrap_or_else(|| "n/a".to_string())
    ));
    out.push_str("\nHuman calibration: pending (blind grading of a gold subset).\n\n");
    out.push_str("## Cost\n\n");
    out.push_str("Computed from logged tokens x list price (input/output tokens from run logs, prices from the published tables).\n\n");
    for row in &report.models {
        let cost = match estimate_usd(&row.id, row.gen_in, row.gen_out) {
            Some(est) => format!("~{} (computed)", usd(est)),
            None => "n/a (provider dashboard)".to_string(),
        };
        out.push_str(&format!(
            "- {}: {} in / {} out tokens, {}\n",
            row.id, row.gen_in, row.gen_out, cost
        ));
    }
    let mut judge_ids: Vec<&String> = report.judge_tokens.keys().collect();
    judge_ids.sort();
    for judge in judge_ids {
        let (input, output) = report.judge_tokens[judge];
        let cost = judge_cost_text(report, judge, "provider dashboard");
        out.push_str(&format!(
            "- judge {judge}: {input} in / {output} out tokens, {cost}\n"
        ));
    }
    if let Some(ramp) = env_usd("MEETING_BENCH_RAMP_USD") {
        out.push_str(&format!("- Ramp Router total (dashboard): {}\n", usd(ramp)));
    }
    if let Some(baseten) = env_usd("MEETING_BENCH_BASETEN_USD") {
        out.push_str(&format!("- Baseten total (dashboard): {}\n", usd(baseten)));
    }
    out.push_str("\n## Per-meeting overall (absolute) and pairwise winner\n\n");
    out.push_str("| Meeting | Winner |");
    for row in &report.models {
        out.push_str(&format!(" {} |", row.id));
    }
    out.push_str("\n| --- | --- |");
    for _ in &report.models {
        out.push_str(" --- |");
    }
    out.push('\n');
    for id in &report.meeting_ids {
        let label = &report.meeting_label[id];
        let winner = report
            .meeting_winner
            .get(id)
            .cloned()
            .unwrap_or_else(|| "-".to_string());
        out.push_str(&format!("| {label} | {winner} |"));
        for row in &report.models {
            let cell = report
                .item_overall
                .get(&(id.clone(), row.id.clone()))
                .map(|score| pct(*score))
                .unwrap_or_else(|| "-".to_string());
            out.push_str(&format!(" {cell} |"));
        }
        out.push('\n');
    }
    out.push_str("\n## Limitations\n\n");
    out.push_str(&format!(
        "- {} of {} judgments ran on inputs clipped to the shared 26k-character judge budget.\n",
        report.truncated_notes, report.judgments
    ));
    out.push_str(&format!(
        "- One sample per meeting and model at provider defaults with reasoning effort {}; reruns vary.\n",
        report.effort
    ));
    out.push_str("- Judges see identical inputs: transcript plus notes clipped to a shared 26k-character budget that fits the smallest judge window (16k tokens).\n");
    out.push_str("- Judges are models; agreement with human preference is validated separately.\n");
    out
}

fn short_id(id: &str) -> &str {
    match id {
        "meta/muse-spark-1.3-contributor" => "muse-spark",
        "google/gemini-3.8-flash" => "gemini-3.8",
        "nvidia/nemotron-3.5-super-vl" => "nemotron-3.5",
        "deepseek-v4.1-flash-1017" => "deepseek-v4.1",
        "claude-haiku-5-5" => "haiku-5.5",
        other => other.rsplit('/').next().unwrap_or(other),
    }
}

/// Cost-vs-quality Pareto frontier as inline SVG (log cost axis).
fn pareto_svg(report: &Report) -> String {
    let meetings = report.meeting_ids.len().max(1) as f64;
    let mut points: Vec<(f64, u32, &str)> = report
        .models
        .iter()
        .filter_map(|row| {
            estimate_usd(&row.id, row.gen_in, row.gen_out)
                .map(|cost| (cost / meetings, row.borda, short_id(&row.id)))
        })
        .collect();
    if points.len() < 2 {
        return String::from("<p>Not enough priced models for a cost frontier.</p>");
    }
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let lo = points
        .iter()
        .map(|p| p.0)
        .fold(f64::INFINITY, f64::min)
        .max(1e-9)
        / 1.35;
    let hi = points.iter().map(|p| p.0).fold(0.0, f64::max).max(lo * 2.0) * 1.35;
    let max_b = points.iter().map(|p| p.1).max().unwrap_or(1) as f64 * 1.1;
    let mut frontier = vec![false; points.len()];
    let mut best = 0u32;
    for (i, p) in points.iter().enumerate() {
        if p.1 > best {
            best = p.1;
            frontier[i] = true;
        }
    }
    let (w, h) = (960.0, 400.0);
    let (pad_l, pad_r, pad_t, pad_b) = (52.0, 132.0, 20.0, 48.0);
    let span = (hi.ln() - lo.ln()).max(1e-9);
    let x = |c: f64| pad_l + (c.ln() - lo.ln()) / span * (w - pad_l - pad_r);
    let y = |b: f64| pad_t + (1.0 - b / max_b) * (h - pad_t - pad_b);
    let mut svg = format!(
        "<svg viewBox=\"0 0 {w} {h}\" width=\"100%\" role=\"img\" aria-label=\"Cost versus quality Pareto frontier\">"
    );
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let value = max_b * frac;
        let yy = y(value);
        svg.push_str(&format!(
            "<line x1=\"{pad_l}\" y1=\"{yy:.1}\" x2=\"{:.1}\" y2=\"{yy:.1}\" stroke=\"#2b2921\" stroke-width=\"1\"/>",
            w - pad_r
        ));
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#8a8474\" font-size=\"11\" text-anchor=\"end\">{:.0}</text>",
            pad_l - 8.0,
            yy + 4.0,
            value
        ));
    }
    for tick in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2] {
        if tick < lo || tick > hi {
            continue;
        }
        let xx = x(tick);
        svg.push_str(&format!(
            "<text x=\"{xx:.1}\" y=\"{:.1}\" fill=\"#8a8474\" font-size=\"11\" text-anchor=\"middle\">${tick}</text>",
            h - pad_b + 20.0
        ));
    }
    svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"#8a8474\" font-size=\"11\" text-anchor=\"middle\">generation cost per note, log scale</text>",
        (pad_l + w - pad_r) / 2.0,
        h - 6.0
    ));
    svg.push_str(&format!(
        "<text x=\"14\" y=\"{:.1}\" fill=\"#8a8474\" font-size=\"11\" text-anchor=\"middle\" transform=\"rotate(-90 14 {:.1})\">Borda points</text>",
        (pad_t + h - pad_b) / 2.0,
        (pad_t + h - pad_b) / 2.0
    ));
    let line_points: Vec<String> = points
        .iter()
        .enumerate()
        .filter(|(i, _)| frontier[*i])
        .map(|(_, p)| format!("{:.1},{:.1}", x(p.0), y(p.1 as f64)))
        .collect();
    svg.push_str(&format!(
        "<polyline class=\"frontier\" points=\"{}\" fill=\"none\" stroke=\"#d9a648\" stroke-width=\"2\"/>",
        line_points.join(" ")
    ));
    for (i, p) in points.iter().enumerate() {
        let (xx, yy) = (x(p.0), y(p.1 as f64));
        let (fill, stroke, label_fill) = if frontier[i] {
            ("#d9a648", "#0f0e0b", "#ece5d3")
        } else {
            ("#4a463c", "#0f0e0b", "#8a8474")
        };
        svg.push_str(&format!(
            "<circle cx=\"{xx:.1}\" cy=\"{yy:.1}\" r=\"6\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"1.5\"/>"
        ));
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"{label_fill}\" font-size=\"12\" font-family=\"ui-monospace, Menlo, monospace\">{}</text>",
            xx + 10.0,
            yy + 4.0,
            p.2
        ));
    }
    svg.push_str("</svg>");
    let frontier_names: Vec<&str> = points
        .iter()
        .enumerate()
        .filter(|(i, _)| frontier[*i])
        .map(|(_, p)| p.2)
        .collect();
    format!(
        "{svg}<p class=\"figcap\">Pareto frontier: {}. No other model is both cheaper and better ranked.</p>",
        frontier_names.join(" → ")
    )
}

fn render_html(report: &Report, date: &str) -> String {
    let max_borda = report
        .models
        .iter()
        .map(|row| row.borda)
        .max()
        .unwrap_or(1)
        .max(1);
    let mut rows = String::new();
    let winner = report.models.first();
    let winner_banner = match winner {
        Some(row) => {
            let wins = report
                .meeting_winner
                .values()
                .filter(|model| *model == &row.id)
                .count();
            format!(
                "<div class=\"winner\"><span class=\"w-rank\">1</span><div><strong>{}</strong> takes the benchmark with {} Borda points, winning {} of {} meetings outright.</div></div>",
                esc(&row.id),
                row.borda,
                wins,
                report.meeting_ids.len()
            )
        }
        None => String::new(),
    };
    for (rank, row) in report.models.iter().enumerate() {
        let width = (row.borda as f64 / max_borda as f64 * 100.0).clamp(0.0, 100.0);
        let medal = if rank < 3 {
            format!("<span class=\"medal m{}\">{}</span>", rank + 1, rank + 1)
        } else {
            format!("<span class=\"ranknum\">{}</span>", rank + 1)
        };
        let rank_cell = |judge: &str| {
            row.judge_ranks
                .get(judge)
                .map(|rank| rank.to_string())
                .unwrap_or_else(|| "-".to_string())
        };
        rows.push_str(&format!(
            "<tr><td>{medal}</td><td class=\"model\">{}</td>\
             <td><div class=\"bar\"><span style=\"width:{width:.1}%\"></span></div></td>\
             <td class=\"num\"><strong>{}</strong></td><td class=\"num\">{}</td>\
             <td class=\"num\">{}</td><td class=\"num\">{}</td>\
             <td class=\"num\">{}</td><td class=\"num\">{}</td></tr>\n",
            esc(&row.id),
            row.borda,
            rank_cell("mammoth"),
            rank_cell("jev"),
            rank_cell("decider"),
            pct(row.pair_pooled),
            pct(row.overall)
        ));
    }
    let mut criteria_head = String::from("<tr><th>Model</th>");
    for (_, label) in CRITERIA {
        criteria_head.push_str(&format!("<th>{label}</th>"));
    }
    criteria_head.push_str("</tr>");
    let mut criteria_rows = String::new();
    for row in &report.models {
        criteria_rows.push_str(&format!("<tr><td class=\"model\">{}</td>", esc(&row.id)));
        for (id, _) in CRITERIA {
            let cell = row
                .criteria
                .get(id)
                .map(|score| pct(*score))
                .unwrap_or_else(|| "-".to_string());
            criteria_rows.push_str(&format!("<td class=\"num\">{cell}</td>"));
        }
        criteria_rows.push_str("</tr>");
    }
    let mut judge_items = String::new();
    for (id, name) in JUDGE_NAMES {
        judge_items.push_str(&format!(
            "<li><strong>{}</strong> <code>{}</code></li>",
            esc(name),
            esc(id)
        ));
    }
    let mut ranking_sections = String::new();
    for (judge, list) in &report.judge_rankings {
        let mut items = String::new();
        for (model, points, pooled) in list.iter() {
            items.push_str(&format!(
                "<li><code>{}</code> — {} pts, P={}</li>",
                esc(model),
                points,
                pct(*pooled)
            ));
        }
        ranking_sections.push_str(&format!("<h3>{}</h3>\n<ol>{}</ol>\n", esc(judge), items));
    }
    let mut agreement_items = String::new();
    for (left, right, rho) in &report.judge_pairs {
        agreement_items.push_str(&format!(
            "<li>{} vs {}: Pearson {}</li>",
            esc(left),
            esc(right),
            rho.map(|r| format!("{r:.2}"))
                .unwrap_or_else(|| "n/a".to_string())
        ));
    }
    agreement_items.push_str(&format!(
        "<li>Mean judge spread per note: {}</li>",
        report
            .spread
            .map(|s| format!("{:.1} pts", s * 100.0))
            .unwrap_or_else(|| "n/a".to_string())
    ));
    let mut cost_rows = String::new();
    for row in &report.models {
        let cost = match estimate_usd(&row.id, row.gen_in, row.gen_out) {
            Some(est) => format!("~{} (computed)", usd(est)),
            None => "n/a (dashboard)".to_string(),
        };
        cost_rows.push_str(&format!(
            "<tr><td class=\"model\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
            esc(&row.id),
            row.gen_in,
            row.gen_out,
            cost
        ));
    }
    let mut judge_cost_ids: Vec<&String> = report.judge_tokens.keys().collect();
    judge_cost_ids.sort();
    for judge in judge_cost_ids {
        let (input, output) = report.judge_tokens[judge];
        let cost = judge_cost_text(report, judge, "dashboard");
        cost_rows.push_str(&format!(
            "<tr><td class=\"model\">judge: {}</td><td class=\"num\">{}</td><td class=\"num\">{}</td><td class=\"num\">{}</td></tr>",
            esc(judge),
            input,
            output,
            cost
        ));
    }
    let mut meeting_details = String::new();
    for id in &report.meeting_ids {
        let label = &report.meeting_label[id];
        let winner = report
            .meeting_winner
            .get(id)
            .cloned()
            .unwrap_or_else(|| "-".to_string());
        let mut items: Vec<(String, f64)> = report
            .models
            .iter()
            .filter_map(|row| {
                report
                    .item_overall
                    .get(&(id.clone(), row.id.clone()))
                    .map(|score| (row.id.clone(), *score))
            })
            .collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut lis = String::new();
        for (model, score) in items {
            lis.push_str(&format!(
                "<li><code>{}</code> — {}</li>",
                esc(&model),
                pct(score)
            ));
        }
        meeting_details.push_str(&format!(
            "<details><summary>{} — pairwise winner: <code>{}</code></summary><ol>{}</ol></details>",
            esc(label),
            esc(&winner),
            lis
        ));
    }
    let truncated_li = format!(
        "<li>{} of {} judgments ran on inputs clipped to the shared 26k-character judge budget.</li>",
        report.truncated_notes, report.judgments
    );
    let effort_li = format!(
        "<li>One sample per meeting and model at provider defaults with reasoning effort {}.</li>",
        esc(&report.effort)
    );

    format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>Meeting-note summarization benchmark</title>\n<style>\n\
         :root {{ color-scheme: dark; --ink: #0f0e0b; --bone: #ece5d3; --muted: #a29a87; \
         --faint: #6b6557; --brass: #d9a648; --line: #26241d; --panel: #141310; }}\n\
         body {{ background: var(--ink); color: var(--bone); \
         font-family: system-ui, -apple-system, sans-serif; max-width: 1000px; \
         margin: 0 auto; padding: 3rem 1.5rem 5rem; line-height: 1.6; \
         background-image: radial-gradient(#ffffff09 1px, transparent 1px); background-size: 22px 22px; }}\n\
         h1, h2, h3 {{ font-family: \"Iowan Old Style\", Palatino, Georgia, serif; letter-spacing: -.01em; }}\n\
         h1 {{ font-size: 2.5rem; line-height: 1.08; margin: .4rem 0 .7rem; font-weight: 600; }}\n\
         h2 {{ font-size: 1.45rem; margin: 3rem 0 .3rem; padding-top: 1.1rem; border-top: 1px solid var(--line); }}\n\
         h3 {{ font-size: 1.1rem; margin: 1.4rem 0 .2rem; }}\n\
         .dateline {{ color: var(--faint); font-size: .85rem; }}\n\
         .dateline b {{ color: var(--brass); font-weight: 600; }}\n\
         .dek {{ color: var(--muted); font-size: 1.02rem; max-width: 64ch; }}\n\
         .winner {{ background: var(--brass); color: #171307; border-radius: 10px; \
         padding: .95rem 1.2rem; margin: 1.6rem 0; display: flex; gap: 1rem; align-items: center; }}\n\
         .winner .w-rank {{ font-family: \"Iowan Old Style\", Palatino, Georgia, serif; \
         font-size: 2.4rem; font-weight: 700; line-height: 1; }}\n\
         .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); margin: 1.4rem 0; \
         border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }}\n\
         .stat {{ padding: .8rem 1rem .8rem 0; }}\n\
         .stat + .stat {{ padding-left: 1rem; border-left: 1px solid var(--line); }}\n\
         .stat b {{ display: block; font-family: \"Iowan Old Style\", Palatino, Georgia, serif; \
         font-size: 1.9rem; font-weight: 600; }}\n\
         .stat span {{ color: var(--muted); font-size: .82rem; }}\n\
         table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}\n\
         th {{ color: var(--muted); font-size: .74rem; text-transform: uppercase; letter-spacing: .07em; \
         font-weight: 600; text-align: right; padding: .5rem .6rem; border-bottom: 1px solid var(--line); }}\n\
         th:first-child, td:first-child {{ text-align: left; }}\n\
         td {{ border-bottom: 1px solid var(--line); padding: .55rem .6rem; text-align: right; \
         font-variant-numeric: tabular-nums; }}\n\
         td.model, td:first-child {{ text-align: left; }}\n\
         tbody tr:hover {{ background: #ffffff08; }}\n\
         td.model {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .86rem; }}\n\
         .ci {{ opacity: .6; font-size: .85em; }}\n\
         .medal {{ display: inline-block; width: 1.55rem; height: 1.55rem; line-height: 1.55rem; \
         text-align: center; border-radius: 50%; font-size: .82rem; font-weight: 700; }}\n\
         .m1 {{ background: var(--brass); color: #171307; }}\n\
         .m2 {{ background: #c9c2b2; color: #171307; }}\n\
         .m3 {{ background: #8a5f36; color: #f5efe0; }}\n\
         .ranknum {{ color: var(--muted); padding-left: .35rem; }}\n\
         .bar {{ background: #ffffff12; border-radius: 3px; height: .65rem; min-width: 110px; }}\n\
         .bar span {{ display: block; height: 100%; border-radius: 3px; background: var(--brass); }}\n\
         .fig {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; \
         padding: 1rem 1rem .4rem; margin: 1rem 0 2rem; }}\n\
         .figcap {{ color: var(--muted); font-size: .85rem; }}\n\
         .frontier {{ stroke-dasharray: 1400; stroke-dashoffset: 1400; \
         animation: draw 1.8s ease-out .3s forwards; }}\n\
         @keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}\n\
         @media (prefers-reduced-motion: reduce) {{ .frontier {{ animation: none; stroke-dashoffset: 0; }} }}\n\
         details {{ border: 1px solid var(--line); border-radius: 8px; padding: .55rem .9rem; margin: .4rem 0; }}\n\
         summary {{ cursor: pointer; font-weight: 600; }}\n\
         code {{ font-size: .88em; }}\nfooter {{ margin-top: 3rem; color: var(--faint); font-size: .9rem; }}\n</style>\n</head>\n<body>\n\
         <div class=\"dateline\"><b>Analog benchmark</b> · {date} · {n_meetings} meetings · {effort} reasoning effort</div>\n\
         <h1>Which LLM writes the best meeting note?</h1>\n\
         <p class=\"dek\">Eight candidates write one note per meeting from the transcript, using the \
         production Analog enhance prompt. Three blind decision-model judges score every note \
         on six criteria and rank every pair of notes; the final leaderboard is Borda over \
         the per-meeting pairwise ranks.</p>\n\
         {winner_banner}\
         <div class=\"stats\">\n\
         <div class=\"stat\"><b>{}</b><span>meetings</span></div>\n\
         <div class=\"stat\"><b>{}</b><span>candidates</span></div>\n\
         <div class=\"stat\"><b>{}</b><span>absolute judgments</span></div>\n\
         <div class=\"stat\"><b>{}</b><span>pairwise judgments</span></div>\n</div>\n\
         <h2>Leaderboard</h2>\n<p class=\"dek\">Borda points over per-meeting pairwise ranks. M / J / D are the \
         Mammoth, Jev, and Decider ballots; Pair P is the mean probability of winning a pairing.</p>\n\
         <table>\n<tr><th>#</th><th>Model</th><th></th>\
         <th>Borda</th><th>M</th><th>J</th><th>D</th><th>Pair P</th><th>Absolute</th></tr>\n{rows}</table>\n\
         <h2>Cost vs quality</h2>\n<p class=\"dek\">Generation cost per note against Borda points. \
         The brass line is the Pareto frontier.</p>\n<figure class=\"fig\">{pareto_svg}</figure>\n\
         <h2>Per-judge rankings</h2>\n{ranking_sections}\
         <h2>Absolute scores by criterion (diagnostic)</h2>\n<table>\n{criteria_head}\n{criteria_rows}</table>\n\
         <h2>Method</h2>\n<ul>{judge_items}</ul>\n\
         <p>Two axes. Absolute: each judge answers one yes/no question per criterion and \
         returns P(yes). Ranking: each judge picks the better note for every pair within each \
         meeting (shown order randomized with a fixed seed). Models rank per meeting per judge; \
         the final order is Borda over all meetings and judges with equal weight per meeting \
         per judge. Model names hide behind blind labels during grading. Each candidate \
         gets the transcript plus session title and participants through the production enhance \
         template (default format, no output template, no memos, no calendar event).</p>\n\
         <h2>Judge agreement</h2>\n<ul>{agreement_items}</ul>\n\
         <p>Human calibration: pending (blind grading of a gold subset).</p>\n\
         <h2>Cost and tokens</h2>\n<table>\n\
         <tr><th>Model</th><th>Input tokens</th><th>Output tokens</th><th>Cost</th></tr>\n{cost_rows}</table>\n\
         <h2>Per meeting (absolute scores + pairwise winner)</h2>\n{meeting_details}\n\
         <h2>Limitations</h2>\n<ul>\n{truncated_li}{effort_li}<li>Reruns vary.</li>\n\
         <li>Judges see identical inputs, clipped to a shared 26k-character budget (smallest judge window).</li>\n\
         <li>Judges are models; agreement with human preference is validated separately.</li>\n</ul>\n\
         <footer>Generated {date} · Scores only; no transcript or note text is published.</footer>\n\
         </body>\n</html>\n",
        report.meeting_ids.len(),
        report.models.len(),
        report.judgments,
        report.pairs,
        n_meetings = report.meeting_ids.len(),
        effort = esc(&report.effort),
        winner_banner = winner_banner,
        pareto_svg = pareto_svg(report),
        rows = rows,
        criteria_head = criteria_head,
        criteria_rows = criteria_rows,
        judge_items = judge_items,
        agreement_items = agreement_items,
        cost_rows = cost_rows,
        meeting_details = meeting_details,
        truncated_li = truncated_li,
        ranking_sections = ranking_sections,
        effort_li = effort_li,
    )
}

fn today() -> String {
    // Days since epoch -> civil date (std only).
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() / 86400)
        .unwrap_or(0) as i64;
    let mut y = 1970;
    let mut remaining = days;
    loop {
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        let year_days = if leap { 366 } else { 365 };
        if remaining < year_days {
            break;
        }
        remaining -= year_days;
        y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let months = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut m = 0;
    while remaining >= months[m] {
        remaining -= months[m];
        m += 1;
    }
    format!("{y:04}-{:02}-{:02}", m + 1, remaining + 1)
}

pub fn render() -> Result<(), String> {
    let run_dir = std::env::var("MEETING_BENCH_RUN_DIR")
        .map_err(|_| "set MEETING_BENCH_RUN_DIR".to_string())?;
    let out_dir = std::env::var("MEETING_BENCH_OUT_DIR").unwrap_or_else(|_| run_dir.clone());
    std::fs::create_dir_all(&out_dir).map_err(|err| format!("mkdir {out_dir}: {err}"))?;
    let report = load(&run_dir)?;
    let date = today();
    let markdown = render_markdown(&report);
    let html = render_html(&report, &date);
    std::fs::write(format!("{out_dir}/report.md"), markdown).map_err(|err| err.to_string())?;
    std::fs::write(format!("{out_dir}/index.html"), html).map_err(|err| err.to_string())?;
    println!(
        "wrote report.md + index.html for {} models x {} meetings",
        report.models.len(),
        report.meeting_ids.len()
    );
    Ok(())
}
