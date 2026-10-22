//! System One decision-model client plus bench scoring. One home for judge logic.
//!
//! The three decision judges (Baseten mammoth, Ramp Router Jev, OpenRouter
//! Decider) speak the same contract: `state` + `questions` in, `noul` answers
//! that read as P(yes) in 0..1. Baseten and Ramp serve `/v1/systemone`,
//! OpenRouter serves `/api/alpha/decisions`.

use std::collections::HashMap;
use std::time::Duration;

// ponytail: Ramp Router sits behind Cloudflare bot defense (403/1010) and
// rejects non-browser user agents; harmless to send everywhere.
pub const BROWSER_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36";

/// Fixed judge rubric shared by all three judges: (criterion id, question).
pub const JUDGE_QUESTIONS: [(&str, &str); 6] = [
    (
        "grounded",
        "Is every factual claim in the meeting note supported by the transcript? Answer yes only if nothing is invented, exaggerated, or contradicted.",
    ),
    (
        "decisions",
        "Does the meeting note capture the key decisions made in the meeting?",
    ),
    (
        "actions",
        "Does the meeting note capture the action items with their owners and deadlines where stated?",
    ),
    (
        "coverage",
        "Does the meeting note cover all important discussion topics without dropping anything critical?",
    ),
    (
        "attribution",
        "Is each statement in the meeting note attributed to the correct speaker?",
    ),
    (
        "concise",
        "Is the meeting note concise and well structured, without filler or repetition?",
    ),
];

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
enum Question<'a> {
    #[serde(rename = "noul")]
    Noul { instructions: &'a str },
    #[serde(rename = "choice")]
    Choice {
        instructions: &'a str,
        criteria: HashMap<&'a str, &'a str>,
    },
}

#[derive(Debug, Clone, serde::Serialize)]
struct DecideRequest<'a> {
    model: &'a str,
    state: &'a str,
    questions: HashMap<&'a str, Question<'a>>,
}

/// Overall pairwise question: which of the two notes is better?
pub const PAIR_QUESTION_ID: &str = "better";
pub const PAIR_INSTRUCTIONS: &str = "Which meeting note is better overall? Judge on faithfulness to the transcript, coverage of decisions and action items, correct attribution, and concision.";

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DecisionUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cost: Option<f64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DecisionAnswer {
    #[serde(rename = "type")]
    pub kind: String,
    pub noul: Option<f64>,
    pub score: Option<f64>,
    pub choice: Option<String>,
    pub confidence: Option<f64>,
    pub probabilities: Option<HashMap<String, f64>>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DecisionResponse {
    pub model: String,
    #[serde(default)]
    pub answers: HashMap<String, DecisionAnswer>,
    pub usage: Option<DecisionUsage>,
}

#[derive(Debug, Clone)]
pub struct SystemOneClient {
    url: String,
    api_key: String,
    http: reqwest::Client,
}

impl SystemOneClient {
    /// `url` is the full decision endpoint: `/v1/systemone` on Baseten and
    /// Ramp Router, `/api/alpha/decisions` on OpenRouter.
    pub fn new(url: impl Into<String>, api_key: impl Into<String>) -> Result<Self, String> {
        let http = reqwest::Client::builder()
            .user_agent(BROWSER_USER_AGENT)
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|err| format!("failed to build http client: {err}"))?;
        Ok(Self {
            url: url.into(),
            api_key: api_key.into(),
            http,
        })
    }

    /// Ask every `JUDGE_QUESTIONS` criterion about `state` in one request.
    pub async fn judge_state(&self, model: &str, state: &str) -> Result<DecisionResponse, String> {
        let questions = JUDGE_QUESTIONS
            .iter()
            .map(|(id, instructions)| (*id, Question::Noul { instructions }))
            .collect();
        self.decide(model, state, questions).await
    }

    /// Ask which of two labeled notes is better overall.
    pub async fn judge_pair(&self, model: &str, state: &str) -> Result<DecisionResponse, String> {
        let questions = HashMap::from([(
            PAIR_QUESTION_ID,
            Question::Choice {
                instructions: PAIR_INSTRUCTIONS,
                criteria: HashMap::from([
                    ("A", "Note A is better overall"),
                    ("B", "Note B is better overall"),
                ]),
            },
        )]);
        self.decide(model, state, questions).await
    }

    async fn decide(
        &self,
        model: &str,
        state: &str,
        questions: HashMap<&str, Question<'_>>,
    ) -> Result<DecisionResponse, String> {
        let body = DecideRequest {
            model,
            state,
            questions,
        };
        let mut attempt = 0;
        loop {
            attempt += 1;
            let response = match self
                .http
                .post(&self.url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&body)
                .send()
                .await
            {
                Ok(response) => response,
                Err(err) => {
                    if attempt >= 3 {
                        return Err(format!("systemone request failed: {err}"));
                    }
                    tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
                    continue;
                }
            };
            let status = response.status().as_u16();
            if response.status().is_success() {
                return response
                    .json()
                    .await
                    .map_err(|err| format!("failed to parse systemone response: {err}"));
            }
            let retryable = status == 429 || status >= 500;
            if !retryable || attempt >= 3 {
                let text = response.text().await.unwrap_or_default();
                let clipped = text.chars().take(400).collect::<String>();
                return Err(format!("systemone HTTP {status}: {clipped}"));
            }
            tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
        }
    }
}

/// Read P(yes) per criterion id. Missing or out-of-range answers are dropped.
pub fn noul_scores(response: &DecisionResponse) -> HashMap<String, f64> {
    response
        .answers
        .iter()
        .filter_map(|(id, answer)| {
            let value = answer.noul?;
            (0.0..=1.0).contains(&value).then(|| (id.clone(), value))
        })
        .collect()
}

/// Read P(A) for the pairwise question (P(B) = 1 - P(A)).
pub fn choice_p_a(response: &DecisionResponse) -> Option<f64> {
    let answer = response.answers.get(PAIR_QUESTION_ID)?;
    if let Some(probs) = &answer.probabilities {
        let a = probs.get("A")?;
        return (0.0..=1.0).contains(a).then_some(*a);
    }
    match answer.choice.as_deref() {
        Some("A") => Some(1.0),
        Some("B") => Some(0.0),
        _ => None,
    }
}

pub fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// Deterministic 95% percentile bootstrap CI (fixed LCG seed, not random).
pub fn bootstrap_ci(values: &[f64], resamples: usize) -> Option<(f64, f64)> {
    if values.is_empty() || resamples == 0 {
        return None;
    }
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut next_index = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize % values.len()
    };
    let mut means = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let sum: f64 = (0..values.len()).map(|_| values[next_index()]).sum();
        means.push(sum / values.len() as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = |q: f64| means[((q * (resamples - 1) as f64).round() as usize).min(resamples - 1)];
    Some((at(0.025), at(0.975)))
}

pub fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let mean_left = mean(left)?;
    let mean_right = mean(right)?;
    let mut num = 0.0;
    let mut den_left = 0.0;
    let mut den_right = 0.0;
    for (l, r) in left.iter().zip(right.iter()) {
        num += (l - mean_left) * (r - mean_right);
        den_left += (l - mean_left).powi(2);
        den_right += (r - mean_right).powi(2);
    }
    let den = (den_left * den_right).sqrt();
    (den > 0.0).then(|| num / den)
}

/// Mean judge spread: average (max - min) across judges per item.
pub fn mean_spread(items: &[Vec<f64>]) -> Option<f64> {
    if items.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for scores in items {
        if scores.len() < 2 {
            continue;
        }
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        sum += max - min;
        count += 1;
    }
    (count > 0).then(|| sum / count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_ci_behave_on_fixed_inputs() {
        assert_eq!(mean(&[]), None);
        assert!((mean(&[0.2, 0.8]).unwrap() - 0.5).abs() < 1e-12);
        let (low, high) = bootstrap_ci(&[0.5; 8], 200).unwrap();
        assert!((low - 0.5).abs() < 1e-12 && (high - 0.5).abs() < 1e-12);
        let (low, high) = bootstrap_ci(&[0.0, 1.0], 400).unwrap();
        assert!(low <= 0.5 && high >= 0.5 && low >= 0.0 && high <= 1.0);
        assert_eq!(bootstrap_ci(&[0.5], 400).unwrap(), (0.5, 0.5));
    }

    #[test]
    fn pearson_and_spread_match_known_values() {
        assert!((pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap() - 1.0).abs() < 1e-12);
        assert!((pearson(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]).unwrap() + 1.0).abs() < 1e-12);
        assert_eq!(pearson(&[1.0], &[1.0]), None);
        assert!((mean_spread(&[vec![0.2, 0.8], vec![0.5, 0.5]]).unwrap() - 0.3).abs() < 1e-12);
        assert_eq!(mean_spread(&[]), None);
    }

    #[test]
    fn choice_p_a_prefers_probabilities_over_hard_choice() {
        let soft: DecisionResponse = serde_json::from_value(serde_json::json!({
            "model": "m",
            "answers": {
                "better": {"type": "choice", "choice": "A", "confidence": 0.6,
                            "probabilities": {"A": 0.7, "B": 0.3}}
            }
        }))
        .unwrap();
        assert_eq!(choice_p_a(&soft), Some(0.7));
        let hard: DecisionResponse = serde_json::from_value(serde_json::json!({
            "model": "m",
            "answers": {"better": {"type": "choice", "choice": "B"}}
        }))
        .unwrap();
        assert_eq!(choice_p_a(&hard), Some(0.0));
        let empty: DecisionResponse = serde_json::from_value(serde_json::json!({
            "model": "m", "answers": {}
        }))
        .unwrap();
        assert_eq!(choice_p_a(&empty), None);
    }

    #[test]
    fn noul_scores_drops_missing_and_wild_values() {
        let response: DecisionResponse = serde_json::from_value(serde_json::json!({
            "model": "m",
            "answers": {
                "grounded": {"type": "noul", "noul": 0.9},
                "broken": {"type": "noul", "noul": 7.0},
                "empty": {"type": "noul", "noul": null}
            }
        }))
        .unwrap();
        assert_eq!(
            noul_scores(&response),
            HashMap::from([("grounded".to_string(), 0.9)])
        );
    }
}
