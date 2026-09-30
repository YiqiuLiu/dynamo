// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! System One decision API (`POST /v1/systemone`), the TypeSafe/Jev wire protocol.
//!
//! A request asks `noul`, `choice` and `score` questions about a state. Each
//! question is rendered as one chat prompt and scored as a one-token completion
//! over its answer labels. The prompt wording, labels, scoring and confidence
//! follow SGLang's `/v1/systemone` route (prompt format version 1), so the same
//! model and chat template give the same answers from either server.

use serde::Serialize;
use serde_json::{Map, Value, json};

/// The documented maximum number of options of a choice question.
pub const MAX_CHOICE_OPTIONS: usize = 255;
/// Options that fit the single-letter labels `A` to `Z`.
pub const MAX_LETTER_LABELS: usize = 26;
/// Levels are labeled `0` to `9`.
pub const MAX_SCORE_LEVELS: usize = 10;
/// Each question is one engine call, so a request fans out this wide at most.
pub const MAX_QUESTIONS: usize = 128;
/// Chat template kwargs that switch reasoning on. Decisions render with them off.
pub const REASONING_TOGGLES: [&str; 2] = ["enable_thinking", "thinking"];
pub const REASONING_START: &str = "<think>";
pub const REASONING_END: &str = "</think>";

/// Fields of the older `/v1/decisions` route that would change the answers if
/// silently honored or ignored, so they are refused by name.
const REFUSED_FIELDS: [&str; 3] = [
    "temperature",
    "prompt_format_version",
    "return_prompt_token_ids",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionKind {
    Noul,
    Choice,
    Score,
}

impl QuestionKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub kind: QuestionKind,
    pub instructions: Option<Value>,
    /// Option names, level indices, or `yes` and `no`, in label order.
    pub names: Vec<String>,
    /// Option descriptions, levels, or the true and false descriptions.
    pub details: Vec<Option<Value>>,
}

impl Question {
    /// Single-token answer labels in candidate order.
    pub fn labels(&self) -> Vec<String> {
        match self.kind {
            QuestionKind::Choice => ('A'..='Z')
                .take(self.names.len())
                .map(String::from)
                .collect(),
            QuestionKind::Noul | QuestionKind::Score => self.names.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SystemOneRequest {
    pub model: String,
    pub state: Value,
    /// Questions in request order, keyed by the caller's ids.
    pub questions: Vec<(String, Question)>,
    /// SGLang extension, for chat templates whose reasoning toggle needs a kwarg.
    pub chat_template_kwargs: Map<String, Value>,
}

/// One entry of a FastAPI-style 422 `detail` list. The offending input is not echoed.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ValidationIssue {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub loc: Vec<Value>,
    pub msg: String,
}

impl ValidationIssue {
    fn new(kind: &'static str, loc: &[Value], msg: impl Into<String>) -> Self {
        Self {
            kind,
            loc: loc.to_vec(),
            msg: msg.into(),
        }
    }
}

fn loc(base: &[Value], key: impl Into<Value>) -> Vec<Value> {
    let mut loc = base.to_vec();
    loc.push(key.into());
    loc
}

pub fn is_blank(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(s) => s.trim().is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        Value::Bool(b) => !b,
        Value::Number(_) => false,
    }
}

/// Text as it appears in the prompt: strings as is, objects and arrays as compact JSON.
pub fn render_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// A string, object or array, which is all the protocol accepts as text.
fn check_text(value: &Value, at: &[Value], issues: &mut Vec<ValidationIssue>) -> bool {
    if matches!(value, Value::String(_) | Value::Object(_) | Value::Array(_)) {
        return true;
    }
    issues.push(ValidationIssue::new(
        "text_type",
        at,
        "Input should be a valid string, object or array",
    ));
    false
}

fn optional_text(
    value: Option<&Value>,
    at: &[Value],
    issues: &mut Vec<ValidationIssue>,
) -> Option<Value> {
    match value {
        None | Some(Value::Null) => None,
        Some(value) => check_text(value, at, issues).then(|| value.clone()),
    }
}

fn forbid_extra(
    object: &Map<String, Value>,
    allowed: &[&str],
    at: &[Value],
    issues: &mut Vec<ValidationIssue>,
) {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            issues.push(ValidationIssue::new(
                "extra_forbidden",
                &loc(at, key.as_str()),
                "Extra inputs are not permitted",
            ));
        }
    }
}

/// Refuse option names that would make the rendered option lines ambiguous.
fn check_option_names<'a>(names: impl Iterator<Item = &'a String>) -> Result<(), String> {
    let mut seen = std::collections::HashSet::new();
    for name in names {
        let key = name.trim().to_lowercase();
        if key.is_empty() {
            return Err("option names must be nonempty".to_string());
        }
        if name
            .chars()
            .any(|c| c.is_control() || c == '\u{2028}' || c == '\u{2029}')
        {
            return Err(format!(
                "option name {name:?} must not contain control or line break characters"
            ));
        }
        if !seen.insert(key) {
            return Err(format!("option name {name:?} repeats another option"));
        }
    }
    Ok(())
}

fn parse_question(
    value: &Value,
    at: &[Value],
    issues: &mut Vec<ValidationIssue>,
) -> Option<Question> {
    let Some(object) = value.as_object() else {
        issues.push(ValidationIssue::new(
            "model_type",
            at,
            "Input should be a valid object",
        ));
        return None;
    };
    let kind = match object.get("type").and_then(Value::as_str) {
        Some("noul") => QuestionKind::Noul,
        Some("choice") => QuestionKind::Choice,
        Some("score") => QuestionKind::Score,
        Some(_) | None if object.contains_key("type") => {
            issues.push(ValidationIssue::new(
                "union_tag_invalid",
                &loc(at, "type"),
                "Input tag should be 'noul', 'choice' or 'score'",
            ));
            return None;
        }
        _ => {
            issues.push(ValidationIssue::new(
                "union_tag_not_found",
                at,
                "Unable to extract tag using discriminator 'type'",
            ));
            return None;
        }
    };
    let at = loc(at, kind.as_str());
    let before = issues.len();
    // A misspelled key would silently ask a different question.
    forbid_extra(object, &["type", "instructions", "criteria"], &at, issues);
    let instructions = optional_text(
        object.get("instructions"),
        &loc(&at, "instructions"),
        issues,
    );
    let criteria = object.get("criteria").filter(|v| !v.is_null());
    let criteria_at = loc(&at, "criteria");

    let (names, details) = match kind {
        QuestionKind::Noul => {
            let mut true_text = None;
            let mut false_text = None;
            match criteria {
                None => {}
                Some(Value::Object(sides)) => {
                    forbid_extra(sides, &["true", "false"], &criteria_at, issues);
                    true_text =
                        optional_text(sides.get("true"), &loc(&criteria_at, "true"), issues);
                    false_text =
                        optional_text(sides.get("false"), &loc(&criteria_at, "false"), issues);
                }
                Some(_) => issues.push(ValidationIssue::new(
                    "model_type",
                    &criteria_at,
                    "Input should be a valid object",
                )),
            }
            let asks_something = [&instructions, &true_text, &false_text]
                .iter()
                .any(|text| text.as_ref().is_some_and(|v| !is_blank(v)));
            if issues.len() == before && !asks_something {
                issues.push(ValidationIssue::new(
                    "value_error",
                    &at,
                    "Value error, a noul question needs instructions or a true or false \
                     description to decide on",
                ));
            }
            (
                vec!["yes".to_string(), "no".to_string()],
                vec![true_text, false_text],
            )
        }
        QuestionKind::Choice => {
            let Some(options) = criteria.and_then(Value::as_object) else {
                issues.push(if criteria.is_none() {
                    ValidationIssue::new("missing", &criteria_at, "Field required")
                } else {
                    ValidationIssue::new(
                        "dict_type",
                        &criteria_at,
                        "Input should be a valid dictionary",
                    )
                });
                return None;
            };
            if options.is_empty() || options.len() > MAX_CHOICE_OPTIONS {
                issues.push(ValidationIssue::new(
                    if options.is_empty() {
                        "too_short"
                    } else {
                        "too_long"
                    },
                    &criteria_at,
                    format!(
                        "Dictionary should have between 1 and {MAX_CHOICE_OPTIONS} items, \
                         not {}",
                        options.len()
                    ),
                ));
            } else if let Err(e) = check_option_names(options.keys()) {
                issues.push(ValidationIssue::new(
                    "value_error",
                    &criteria_at,
                    format!("Value error, {e}"),
                ));
            }
            let details = options
                .iter()
                .map(|(name, description)| {
                    optional_text(Some(description), &loc(&criteria_at, name.as_str()), issues)
                })
                .collect();
            (options.keys().cloned().collect(), details)
        }
        QuestionKind::Score => {
            let Some(levels) = criteria.and_then(Value::as_array) else {
                issues.push(if criteria.is_none() {
                    ValidationIssue::new("missing", &criteria_at, "Field required")
                } else {
                    ValidationIssue::new("list_type", &criteria_at, "Input should be a valid list")
                });
                return None;
            };
            if levels.is_empty() || levels.len() > MAX_SCORE_LEVELS {
                issues.push(ValidationIssue::new(
                    if levels.is_empty() {
                        "too_short"
                    } else {
                        "too_long"
                    },
                    &criteria_at,
                    format!(
                        "List should have between 1 and {MAX_SCORE_LEVELS} items, not {}",
                        levels.len()
                    ),
                ));
            }
            let mut details = Vec::with_capacity(levels.len());
            for (i, level) in levels.iter().enumerate() {
                let level_at = loc(&criteria_at, i);
                if check_text(level, &level_at, issues) && is_blank(level) {
                    issues.push(ValidationIssue::new(
                        "value_error",
                        &level_at,
                        "Value error, must not be blank",
                    ));
                }
                details.push(Some(level.clone()));
            }
            ((0..levels.len()).map(|i| i.to_string()).collect(), details)
        }
    };
    (issues.len() == before).then_some(Question {
        kind,
        instructions,
        names,
        details,
    })
}

/// Validate a request body, reporting every problem the way FastAPI would.
pub fn parse_request(body: &Value) -> Result<SystemOneRequest, Vec<ValidationIssue>> {
    let body_at = [json!("body")];
    let Some(object) = body.as_object() else {
        return Err(vec![ValidationIssue::new(
            "model_attributes_type",
            &body_at,
            "Input should be a valid object",
        )]);
    };
    let mut issues = Vec::new();

    let model = match object.get("model") {
        Some(Value::String(model)) => model.clone(),
        Some(_) => {
            issues.push(ValidationIssue::new(
                "string_type",
                &loc(&body_at, "model"),
                "Input should be a valid string",
            ));
            String::new()
        }
        None => {
            issues.push(ValidationIssue::new(
                "missing",
                &loc(&body_at, "model"),
                "Field required",
            ));
            String::new()
        }
    };

    let state = match object.get("state") {
        Some(state) => {
            check_text(state, &loc(&body_at, "state"), &mut issues);
            state.clone()
        }
        None => {
            issues.push(ValidationIssue::new(
                "missing",
                &loc(&body_at, "state"),
                "Field required",
            ));
            Value::Null
        }
    };

    let mut questions = Vec::new();
    let questions_at = loc(&body_at, "questions");
    match object.get("questions") {
        Some(Value::Object(map)) if !map.is_empty() => {
            for (id, question) in map {
                if let Some(question) =
                    parse_question(question, &loc(&questions_at, id.as_str()), &mut issues)
                {
                    questions.push((id.clone(), question));
                }
            }
        }
        Some(Value::Object(_)) => issues.push(ValidationIssue::new(
            "too_short",
            &questions_at,
            "Dictionary should have at least 1 item after validation, not 0",
        )),
        Some(_) => issues.push(ValidationIssue::new(
            "dict_type",
            &questions_at,
            "Input should be a valid dictionary",
        )),
        None => issues.push(ValidationIssue::new(
            "missing",
            &questions_at,
            "Field required",
        )),
    }

    let chat_template_kwargs = match object.get("chat_template_kwargs") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(kwargs)) => kwargs.clone(),
        Some(_) => {
            issues.push(ValidationIssue::new(
                "dict_type",
                &loc(&body_at, "chat_template_kwargs"),
                "Input should be a valid dictionary",
            ));
            Map::new()
        }
    };

    for field in REFUSED_FIELDS {
        if object.get(field).is_some_and(|v| !v.is_null()) {
            issues.push(ValidationIssue::new(
                "value_error",
                &loc(&body_at, field),
                format!("Value error, {field} is not part of this API, use /v1/decisions for it"),
            ));
        }
    }

    if issues.is_empty() {
        Ok(SystemOneRequest {
            model,
            state,
            questions,
            chat_template_kwargs,
        })
    } else {
        Err(issues)
    }
}

/// The user message of one question, in the wording of prompt format version 1.
pub fn render_question(state: &str, question: &Question, labels: &[String]) -> String {
    let question_text = match &question.instructions {
        Some(text) if !is_blank(text) => render_text(Some(text)),
        _ => String::new(),
    };
    let mut lines = Vec::new();
    match question.kind {
        QuestionKind::Choice => {
            if !question_text.is_empty() {
                lines.push(format!("Question: {question_text}"));
            }
            for ((label, name), description) in
                labels.iter().zip(&question.names).zip(&question.details)
            {
                let detail = render_text(description.as_ref());
                lines.push(if detail.is_empty() {
                    format!("{label}: {name}")
                } else {
                    format!("{label}: {name} - {detail}")
                });
            }
            lines.push("Answer with the letter of one option only.".to_string());
        }
        QuestionKind::Score => {
            if !question_text.is_empty() {
                lines.push(format!("Question: {question_text}"));
            }
            for (label, level) in labels.iter().zip(&question.details) {
                lines.push(format!("{label}: {}", render_text(level.as_ref())));
            }
            lines.push("Answer with the number of one level only.".to_string());
        }
        QuestionKind::Noul => {
            lines.push(if question_text.is_empty() {
                "Is the following true?".to_string()
            } else {
                format!("Is the following true? {question_text}")
            });
            for (label, description) in labels.iter().zip(&question.details) {
                let detail = render_text(description.as_ref());
                if !detail.is_empty() {
                    lines.push(format!("{label}: {detail}"));
                }
            }
            lines.push("Answer with yes or no only.".to_string());
        }
    }
    let mut out = String::with_capacity(state.len() + 256);
    out.push_str(state);
    out.push_str("\n\n");
    out.push_str(&lines.join("\n"));
    out
}

/// Chat template kwargs for rendering: reasoning off, then the request's own.
/// Fails when the request turns a reasoning toggle on.
pub fn chat_template_kwargs(
    request_kwargs: &Map<String, Value>,
) -> Result<Map<String, Value>, String> {
    let mut kwargs = Map::new();
    for toggle in REASONING_TOGGLES {
        if let Some(value) = request_kwargs.get(toggle)
            && value != &Value::Bool(false)
        {
            return Err(format!(
                "chat_template_kwargs sets {toggle:?} to {value}, but decisions need it false or unset"
            ));
        }
        kwargs.insert(toggle.to_string(), Value::Bool(false));
    }
    for (key, value) in request_kwargs {
        kwargs.insert(key.clone(), value.clone());
    }
    Ok(kwargs)
}

/// Whether the rendered prompt leaves a reasoning block open at the answer position.
/// Only the generation prompt after the message's fixed last line is inspected.
pub fn leaves_reasoning_open(prompt: &str, message: &str) -> bool {
    let closing = message.rsplit('\n').next().unwrap_or(message);
    let generation_prompt = match prompt.rfind(closing) {
        Some(cut) => &prompt[cut + closing.len()..],
        None => prompt,
    };
    match (
        generation_prompt.rfind(REASONING_START),
        generation_prompt.rfind(REASONING_END),
    ) {
        (Some(opened), Some(closed)) => opened > closed,
        (Some(_), None) => true,
        _ => false,
    }
}

/// Softmax over the raw label logprobs, which equals SGLang's softmax over the
/// label logits at temperature 1.
pub fn softmax(logprobs: &[f64]) -> Vec<f64> {
    let max = logprobs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logprobs.iter().map(|lp| (lp - max).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / total).collect()
}

/// Full-vocabulary probability that landed on the offered labels.
pub fn label_mass(logprobs: &[f64]) -> f64 {
    logprobs.iter().map(|lp| lp.exp()).sum()
}

/// How far the top option stands above a uniform guess, from 0 to 1.
pub fn choice_confidence(q: &[f64]) -> f64 {
    let n = q.len() as f64;
    if q.len() == 1 {
        return 1.0;
    }
    let top = q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ((n * top - 1.0) / (n - 1.0)).clamp(0.0, 1.0)
}

/// One minus the spread around the top level relative to a uniform spread,
/// floored at 0 (the formula of typesafe-ai/system-one-adapter-python).
pub fn score_confidence(q: &[f64]) -> f64 {
    let n = q.len();
    if n == 1 {
        return 1.0;
    }
    let top = argmax(q);
    let spread: f64 = q
        .iter()
        .enumerate()
        .map(|(i, p)| p * (i as f64 - top as f64).abs())
        .sum();
    let center = (n as f64 - 1.0) / 2.0;
    let uniform_spread: f64 = (0..n).map(|i| (i as f64 - center).abs()).sum::<f64>() / n as f64;
    (1.0 - spread / uniform_spread).max(0.0)
}

/// Index of the largest value, ties going to the first.
fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

fn normalized(probabilities: &[f64]) -> Vec<f64> {
    let total: f64 = probabilities.iter().sum();
    if total <= 0.0 {
        return vec![1.0 / probabilities.len() as f64; probabilities.len()];
    }
    probabilities.iter().map(|p| p / total).collect()
}

/// The answer to one question from its raw label logprobs, in label order.
/// Fails on non-finite values, which are a server fault rather than a client error.
pub fn answer(question: &Question, label_logprobs: &[f64]) -> Result<Value, String> {
    let probabilities = softmax(label_logprobs);
    let mass = label_mass(label_logprobs);
    if !probabilities.iter().chain([&mass]).all(|v| v.is_finite()) {
        return Err("scored non-finite values".to_string());
    }
    let by_name = |names: &[String]| -> Map<String, Value> {
        names
            .iter()
            .zip(&probabilities)
            .map(|(name, p)| (name.clone(), json!(p)))
            .collect()
    };
    Ok(match question.kind {
        QuestionKind::Noul => json!({
            "type": "noul",
            "noul": probabilities[0],
            "x_label_mass": mass,
        }),
        QuestionKind::Choice => json!({
            "type": "choice",
            "choice": question.names[argmax(&probabilities)],
            "confidence": choice_confidence(&normalized(&probabilities)),
            "probabilities": by_name(&question.names),
            "x_label_mass": mass,
        }),
        QuestionKind::Score => {
            let legend: Map<String, Value> = question
                .names
                .iter()
                .zip(&question.details)
                .map(|(name, level)| (name.clone(), level.clone().unwrap_or(Value::Null)))
                .collect();
            json!({
                "type": "score",
                "score": probabilities.iter().enumerate().map(|(i, p)| i as f64 * p).sum::<f64>(),
                "confidence": score_confidence(&normalized(&probabilities)),
                "legend": legend,
                "probabilities": by_name(&question.names),
                "x_label_mass": mass,
            })
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(body: Value) -> SystemOneRequest {
        parse_request(&body).expect("valid request")
    }

    fn issues(body: Value) -> Vec<ValidationIssue> {
        parse_request(&body).expect_err("invalid request")
    }

    #[test]
    fn parses_all_three_question_types_in_order() {
        let req = request(json!({
            "model": "m",
            "state": {"ticket": "down"},
            "questions": {
                "urgent": {"type": "noul", "instructions": "Reply within the hour?"},
                "team": {"type": "choice", "criteria": {"billing": null, "infra": "servers"}},
                "severity": {"type": "score", "criteria": ["low", "mid", "high"]},
            },
            "unknown_top_level": 1,
        }));
        let ids: Vec<_> = req.questions.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["urgent", "team", "severity"]);
        assert_eq!(req.questions[1].1.names, ["billing", "infra"]);
        assert_eq!(req.questions[1].1.labels(), ["A", "B"]);
        assert_eq!(req.questions[2].1.labels(), ["0", "1", "2"]);
        assert_eq!(req.questions[0].1.labels(), ["yes", "no"]);
    }

    #[test]
    fn rejects_unknown_question_keys_and_refused_fields() {
        let found = issues(json!({
            "model": "m",
            "state": "s",
            "questions": {"q": {"type": "noul", "instruction": "typo"}},
            "temperature": 0.5,
        }));
        let kinds: Vec<_> = found.iter().map(|i| (i.kind, i.loc.clone())).collect();
        assert!(kinds.contains(&(
            "extra_forbidden",
            vec![
                json!("body"),
                json!("questions"),
                json!("q"),
                json!("noul"),
                json!("instruction")
            ]
        )));
        assert!(kinds.contains(&("value_error", vec![json!("body"), json!("temperature")])));
    }

    #[test]
    fn rejects_empty_and_duplicate_criteria() {
        let found = issues(json!({
            "model": "m",
            "state": "s",
            "questions": {
                "a": {"type": "choice", "criteria": {}},
                "b": {"type": "choice", "criteria": {"Yes": null, " yes": null}},
                "c": {"type": "score", "criteria": ["ok", " "]},
                "d": {"type": "noul"},
            },
        }));
        assert_eq!(found.len(), 4, "{found:?}");
    }

    #[test]
    fn requires_model_state_and_questions() {
        let found = issues(json!({}));
        let fields: Vec<_> = found.iter().map(|i| i.loc[1].clone()).collect();
        assert_eq!(fields, [json!("model"), json!("state"), json!("questions")]);
        assert!(found.iter().all(|i| i.kind == "missing"));
    }

    #[test]
    fn renders_the_prompt_format_version_1_wording() {
        let req = request(json!({
            "model": "m",
            "state": {"ticket": "Everything is down"},
            "questions": {
                "team": {"type": "choice", "instructions": "Which team?",
                         "criteria": {"billing": null, "infra": "servers"}},
                "urgent": {"type": "noul", "instructions": "Urgent?",
                           "criteria": {"true": "within the hour"}},
                "severity": {"type": "score", "criteria": ["low", "high"]},
            },
        }));
        let state = render_text(Some(&req.state));
        assert_eq!(state, r#"{"ticket":"Everything is down"}"#);
        let render = |i: usize| {
            let q = &req.questions[i].1;
            render_question(&state, q, &q.labels())
        };
        assert_eq!(
            render(0),
            format!(
                "{state}\n\nQuestion: Which team?\nA: billing\nB: infra - servers\n\
                 Answer with the letter of one option only."
            )
        );
        assert_eq!(
            render(1),
            format!(
                "{state}\n\nIs the following true? Urgent?\nyes: within the hour\n\
                 Answer with yes or no only."
            )
        );
        assert_eq!(
            render(2),
            format!("{state}\n\n0: low\n1: high\nAnswer with the number of one level only.")
        );
    }

    #[test]
    fn forces_reasoning_off_and_refuses_it_on() {
        let kwargs = chat_template_kwargs(&Map::new()).unwrap();
        assert_eq!(kwargs["enable_thinking"], json!(false));
        let mut on = Map::new();
        on.insert("enable_thinking".into(), json!(true));
        assert!(chat_template_kwargs(&on).is_err());
    }

    #[test]
    fn detects_open_reasoning_after_the_message() {
        let msg = "state <think>\n\nAnswer with yes or no only.";
        let closed = format!(
            "<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let open = format!("<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n<think>\n");
        assert!(!leaves_reasoning_open(&closed, msg));
        assert!(leaves_reasoning_open(&open, msg));
        assert!(!leaves_reasoning_open("user: x\nassistant:", "x"));
    }

    #[test]
    fn confidence_matches_the_published_examples() {
        let p = [0.0, 0.95, 0.05];
        assert!((score_confidence(&p) - 0.925).abs() < 1e-9);
        assert!((choice_confidence(&p) - 0.925).abs() < 1e-9);
        assert_eq!(score_confidence(&[0.5, 0.0, 0.5]), 0.0);
        assert_eq!(choice_confidence(&[1.0]), 1.0);
    }

    #[test]
    fn answers_are_scored_from_raw_logprobs() {
        let q = Question {
            kind: QuestionKind::Score,
            instructions: None,
            names: vec!["0".into(), "1".into()],
            details: vec![Some(json!("low")), Some(json!("high"))],
        };
        let lp = [0.25f64.ln(), 0.25f64.ln()];
        let a = answer(&q, &lp).unwrap();
        assert!((a["score"].as_f64().unwrap() - 0.5).abs() < 1e-12);
        assert!((a["x_label_mass"].as_f64().unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(a["legend"], json!({"0": "low", "1": "high"}));
        let keys: Vec<_> = a.as_object().unwrap().keys().cloned().collect();
        assert_eq!(
            keys,
            [
                "type",
                "score",
                "confidence",
                "legend",
                "probabilities",
                "x_label_mass"
            ]
        );
        assert!(answer(&q, &[f64::NEG_INFINITY, f64::NEG_INFINITY]).is_err());
    }

    #[test]
    fn choice_ties_go_to_the_first_option() {
        let q = Question {
            kind: QuestionKind::Choice,
            instructions: None,
            names: vec!["x".into(), "y".into()],
            details: vec![None, None],
        };
        let a = answer(&q, &[-1.0, -1.0]).unwrap();
        assert_eq!(a["choice"], "x");
        assert_eq!(a["confidence"], json!(0.0));
    }
}
