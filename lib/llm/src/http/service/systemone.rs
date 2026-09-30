// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `POST /v1/systemone`: System One decisions over the completions engine.
//!
//! Every question becomes a token-ID completion with `max_tokens=1` whose
//! `logprob_token_ids` are the answer labels. The first question goes through
//! the router alone so one worker computes the shared state prefix; the rest are
//! pinned to that worker, because the router's view of its cache lags behind a
//! request that just finished.

use std::sync::{Arc, LazyLock};

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use dashmap::DashMap;
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use serde_json::{Value, json};
use tracing::Instrument;

use dynamo_renderer::{OAIChatLikeRequest, PromptFormatter};

use super::{
    RouteDoc,
    disconnect::create_connection_monitor,
    metrics::{CancellationLabels, Endpoint, InflightGuard},
    openai::{
        ErrorMessage, ErrorResponse, check_for_backend_error, check_model_serving_ready,
        check_ready, context_from_headers, copy_context_metadata, extract_error_type_from_response,
        get_body_limit, get_or_create_request_id, non_streaming_aggregation_error_response,
        read_json_request_body,
    },
    service_v2::{self, BackendErrorCheck},
};
use crate::model_card::ModelDeploymentCard;
use crate::preprocessor::prompt::prompt_formatter_from_mdc;
use crate::protocols::openai::ParsingOptions;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::protocols::openai::completions::{
    NvCreateCompletionRequest, NvCreateCompletionResponse,
};
use crate::protocols::systemone::{self, Question, SystemOneRequest};
use crate::tokenizers::Tokenizer;
use crate::types::openai::completions::OpenAICompletionsStreamingEngine;

pub(crate) const DEFAULT_SYSTEMONE_PATH: &str = "/v1/systemone";

pub fn systemone_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| DEFAULT_SYSTEMONE_PATH.to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(handler_systemone))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc], router)
}

/// What a served model needs to render and label questions, built once per card.
struct ModelContext {
    mdcsum: String,
    tokenizer: Tokenizer,
    formatter: PromptFormatter,
    context_length: u32,
}

static MODEL_CONTEXTS: LazyLock<DashMap<String, Arc<ModelContext>>> = LazyLock::new(DashMap::new);

async fn model_context(card: &ModelDeploymentCard) -> anyhow::Result<Arc<ModelContext>> {
    if let Some(cached) = MODEL_CONTEXTS.get(&card.display_name)
        && cached.mdcsum == card.mdcsum()
    {
        return Ok(cached.clone());
    }
    // Catalog cards can still point at the worker's metadata URLs; resolve them
    // through the local MDC cache, which the watcher has already filled.
    let mdcsum = card.mdcsum().to_string();
    let mut card = card.clone();
    card.download_config(None).await?;
    let context = Arc::new(ModelContext {
        mdcsum,
        tokenizer: card.tokenizer()?,
        formatter: prompt_formatter_from_mdc(&card)?,
        context_length: card.effective_context_length(),
    });
    MODEL_CONTEXTS.insert(card.display_name.clone(), context.clone());
    Ok(context)
}

/// A question ready to score: its prompt token ids and one token id per label.
struct EncodedQuestion {
    prompt: Vec<u32>,
    labels: Vec<u32>,
}

/// Answers to one question from the engine.
struct Scored {
    label_logprobs: Vec<f64>,
    prompt_tokens: u32,
    worker_id: Option<u64>,
}

fn validation_error(issues: Vec<systemone::ValidationIssue>) -> Response {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(json!({ "detail": issues })),
    )
        .into_response()
}

/// Requests the model or its chat template cannot answer faithfully.
fn unsupported(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({ "detail": message.into() })),
    )
        .into_response()
}

async fn handler_systemone(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let body = match read_json_request_body(&headers, body).await {
        Ok(body) => body,
        Err(e) => return e.into_response(),
    };
    let request = match serde_json::from_slice::<Value>(&body) {
        Ok(value) => systemone::parse_request(&value),
        Err(e) => Err(vec![systemone::ValidationIssue {
            kind: "json_invalid",
            loc: vec![json!("body"), json!(e.column())],
            msg: format!("JSON decode error: {e}"),
        }]),
    };
    let mut request = match request {
        Ok(request) => request,
        Err(issues) => return validation_error(issues),
    };
    if request.questions.len() > systemone::MAX_QUESTIONS {
        return unsupported(format!(
            "the request has {} questions, this server answers at most {}",
            request.questions.len(),
            systemone::MAX_QUESTIONS
        ));
    }
    if let Err(e) = check_ready(&state) {
        return e.into_response();
    }
    request.model = state.manager().resolve_canonical_name(&request.model);
    if let Err(e) = check_model_serving_ready(&state, &request.model) {
        return e.into_response();
    }

    let request_id = get_or_create_request_id(&headers);
    let metric_model = state.manager().metric_model_for(&request.model).to_string();
    let parent = match context_from_headers((), request_id.clone(), &headers) {
        Ok(parent) => parent,
        Err(e) => return e.into_response(),
    };
    let (mut connection_handle, _stream_handle) = create_connection_monitor(
        parent.context(),
        Some(state.metrics_clone()),
        CancellationLabels {
            model: metric_model.clone(),
            endpoint: Endpoint::SystemOne.to_string(),
            request_type: "unary".to_string(),
        },
    )
    .await;
    let inflight_guard = state.metrics_clone().create_inflight_guard(
        &metric_model,
        Endpoint::SystemOne,
        false,
        &request_id,
    );

    let response =
        tokio::spawn(systemone(state, request, parent, inflight_guard).in_current_span())
            .await
            .unwrap_or_else(|e| {
                ErrorMessage::internal_server_error_with_details(
                    "Failed to await System One task",
                    format!("{e:?}"),
                )
                .into_response()
            });
    connection_handle.disarm();
    response
}

async fn systemone(
    state: Arc<service_v2::State>,
    request: SystemOneRequest,
    parent: Context<()>,
    mut inflight_guard: InflightGuard,
) -> Response {
    let fail = |guard: &mut InflightGuard, e: ErrorResponse| {
        guard.mark_error(extract_error_type_from_response(&e));
        e.into_response()
    };
    let (engine, parsing_options) = match state
        .manager()
        .get_completions_engine_with_parsing(&request.model)
    {
        Ok(found) => found,
        Err(e) => return fail(&mut inflight_guard, ErrorMessage::from_model_error(&e)),
    };
    let Some(card) = state.manager().get_model_card_by_name(&request.model) else {
        return fail(&mut inflight_guard, ErrorMessage::model_not_found());
    };
    let context = match model_context(&card).await {
        Ok(context) => context,
        Err(e) => {
            return fail(
                &mut inflight_guard,
                ErrorMessage::internal_server_error_with_details(
                    "Failed to load the model tokenizer or chat template",
                    format!("{e:#}"),
                ),
            );
        }
    };

    // Rendering and tokenizing every question is CPU work on the whole state.
    let encode_request = request.clone();
    let encoded =
        match tokio::task::spawn_blocking(move || encode_all(&context, &encode_request)).await {
            Ok(Ok(encoded)) => encoded,
            Ok(Err(message)) => {
                inflight_guard.mark_error(super::metrics::ErrorType::Validation);
                return unsupported(message);
            }
            Err(e) => {
                return fail(
                    &mut inflight_guard,
                    ErrorMessage::internal_server_error_with_details(
                        "Failed to encode System One questions",
                        format!("{e:?}"),
                    ),
                );
            }
        };

    let scorer = Scorer {
        engine,
        parsing_options,
        model: request.model.clone(),
        parent: &parent,
    };
    let mut encoded = encoded.into_iter().enumerate();
    let (first_index, first) = encoded
        .next()
        .expect("requests carry at least one question");
    let first = match scorer.score(first_index, first, None).await {
        Ok(scored) => scored,
        Err(e) => {
            parent.context().kill();
            return fail(&mut inflight_guard, e);
        }
    };
    let pinned = first.worker_id;
    let rest = futures::future::try_join_all(
        encoded.map(|(index, question)| scorer.score(index, question, pinned)),
    )
    .await;
    let rest = match rest {
        Ok(rest) => rest,
        Err(e) => {
            // One failed question fails the request, so stop the others.
            parent.context().kill();
            return fail(&mut inflight_guard, e);
        }
    };

    let mut answers = serde_json::Map::new();
    let mut input_tokens: u64 = 0;
    for ((question_id, question), scored) in request
        .questions
        .iter()
        .zip([first].into_iter().chain(rest))
    {
        input_tokens += u64::from(scored.prompt_tokens);
        match systemone::answer(question, &scored.label_logprobs) {
            Ok(answer) => {
                answers.insert(question_id.clone(), answer);
            }
            Err(e) => {
                return fail(
                    &mut inflight_guard,
                    ErrorMessage::internal_server_error(&format!("question {question_id:?} {e}")),
                );
            }
        }
    }
    if parent.context().is_killed() {
        inflight_guard.mark_error(super::metrics::ErrorType::Cancelled);
    } else {
        inflight_guard.mark_ok();
    }
    Json(json!({
        "model": request.model,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }))
    .into_response()
}

fn render(
    context: &ModelContext,
    request: &NvCreateChatCompletionRequest,
) -> anyhow::Result<String> {
    match &context.formatter {
        PromptFormatter::OAI(formatter) => formatter.render(request as &dyn OAIChatLikeRequest),
    }
}

fn encode_all(
    context: &ModelContext,
    request: &SystemOneRequest,
) -> Result<Vec<EncodedQuestion>, String> {
    let kwargs = systemone::chat_template_kwargs(&request.chat_template_kwargs)?;
    let state = systemone::render_text(Some(&request.state));
    request
        .questions
        .iter()
        .map(|(question_id, question)| {
            encode_question(context, &request.model, &state, question, &kwargs)
                .map_err(|e| format!("question {question_id:?}: {e}"))
        })
        .collect()
}

fn encode_question(
    context: &ModelContext,
    model: &str,
    state: &str,
    question: &Question,
    kwargs: &serde_json::Map<String, Value>,
) -> Result<EncodedQuestion, String> {
    if question.names.len() > systemone::MAX_LETTER_LABELS {
        return Err(format!(
            "it has {} options, and this server labels at most {} (A to Z)",
            question.names.len(),
            systemone::MAX_LETTER_LABELS
        ));
    }
    let labels = question.labels();
    let message = systemone::render_question(state, question, &labels);
    let chat_request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "chat_template_kwargs": kwargs,
    }))
    .map_err(|e| format!("could not build the chat prompt: {e}"))?;
    let prompt =
        render(context, &chat_request).map_err(|e| format!("the chat template failed: {e}"))?;
    if systemone::leaves_reasoning_open(&prompt, &message) {
        return Err(
            "the chat template leaves a reasoning block open at the answer position, so this \
             model is not supported with these chat_template_kwargs"
                .to_string(),
        );
    }
    let encode = |text: &str| -> Result<Vec<u32>, String> {
        context
            .tokenizer
            .encode(text)
            .map(|encoding| encoding.token_ids().to_vec())
            .map_err(|e| format!("tokenization failed: {e}"))
    };
    let prompt_ids = encode(&prompt)?;
    if prompt_ids.len() + 1 > context.context_length as usize {
        return Err(format!(
            "the prompt has {} tokens, which does not fit the context length of {} tokens",
            prompt_ids.len(),
            context.context_length
        ));
    }
    // A label is usable only if it adds exactly one distinct token after the prompt.
    // A two-token label would read the wrong logit without any error.
    let mut label_ids = Vec::with_capacity(labels.len());
    for label in &labels {
        let ids = encode(&format!("{prompt}{label}"))?;
        let token = (ids.len() == prompt_ids.len() + 1
            && ids[..prompt_ids.len()] == prompt_ids[..])
            .then(|| ids[prompt_ids.len()])
            .filter(|id| !label_ids.contains(id));
        match token {
            Some(id) => label_ids.push(id),
            None => {
                return Err(format!(
                    "the answer label {label:?} is not one distinct token after the chat prompt \
                     for this tokenizer, so this model is not supported"
                ));
            }
        }
    }
    Ok(EncodedQuestion {
        prompt: prompt_ids,
        labels: label_ids,
    })
}

struct Scorer<'a> {
    engine: OpenAICompletionsStreamingEngine,
    parsing_options: ParsingOptions,
    model: String,
    parent: &'a Context<()>,
}

impl Scorer<'_> {
    async fn score(
        &self,
        index: usize,
        question: EncodedQuestion,
        pinned_worker: Option<u64>,
    ) -> Result<Scored, ErrorResponse> {
        let mut nvext = json!({ "extra_fields": ["worker_id"] });
        if let Some(worker_id) = pinned_worker {
            nvext["backend_instance_id"] = json!(worker_id);
        }
        // `logprobs` only switches logprobs on in the response; the worker replaces
        // the top-k width with the label ids.
        let completion: NvCreateCompletionRequest = serde_json::from_value(json!({
            "model": self.model,
            "prompt": question.prompt,
            "max_tokens": 1,
            "temperature": 1.0,
            "logprobs": 1,
            "return_tokens_as_token_ids": true,
            "logprob_token_ids": question.labels,
            "nvext": nvext,
        }))
        .map_err(|e| {
            ErrorMessage::internal_server_error_with_details(
                "Failed to build the scoring request",
                e.to_string(),
            )
        })?;
        let mut child = Context::with_id_and_metadata(
            completion,
            format!("{}-{index}", self.parent.id()),
            self.parent.metadata().clone(),
        );
        copy_context_metadata(self.parent, &mut child);

        let stream =
            self.engine.generate(child).await.map_err(|e| {
                ErrorMessage::from_anyhow(e, "Failed to score a System One question")
            })?;
        self.parent.context().link_child(stream.context());
        let stream = check_for_backend_error(stream, BackendErrorCheck::UntilFirstEvent).await?;
        let response =
            NvCreateCompletionResponse::from_annotated_stream(stream, self.parsing_options.clone())
                .await
                .map_err(|e| {
                    non_streaming_aggregation_error_response(e, "Failed to fold a scoring response")
                })?;
        read_scores(&response, &question.labels)
    }
}

/// The raw logprob of every label, read from the `token_id:<id>` keyed top logprobs.
fn read_scores(
    response: &NvCreateCompletionResponse,
    labels: &[u32],
) -> Result<Scored, ErrorResponse> {
    let top = response
        .inner
        .choices
        .first()
        .and_then(|choice| choice.logprobs.as_ref())
        .and_then(|logprobs| logprobs.top_logprobs.first())
        .and_then(Value::as_array);
    let mut by_id = std::collections::HashMap::new();
    for entry in top.into_iter().flatten() {
        let id = entry
            .get("token")
            .and_then(Value::as_str)
            .and_then(|token| token.strip_prefix("token_id:"))
            .and_then(|id| id.parse::<u32>().ok());
        if let (Some(id), Some(logprob)) = (id, entry.get("logprob").and_then(Value::as_f64)) {
            by_id.insert(id, logprob);
        }
    }
    let label_logprobs = labels
        .iter()
        .map(|id| by_id.get(id).copied())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            // Seen when the worker ignores logprob_token_ids, e.g. under speculative decoding.
            ErrorMessage::internal_server_error_with_details(
                "The worker did not return logprobs for every answer label",
                format!("labels {labels:?}, returned {by_id:?}"),
            )
        })?;
    let worker_id = response
        .nvext
        .as_ref()
        .and_then(|nvext| nvext.pointer("/worker_id/decode_worker_id"))
        .and_then(Value::as_u64);
    Ok(Scored {
        label_logprobs,
        prompt_tokens: response
            .inner
            .usage
            .as_ref()
            .map_or(0, |usage| usage.prompt_tokens),
        worker_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(top_logprobs: Value, nvext: Option<Value>) -> NvCreateCompletionResponse {
        serde_json::from_value(json!({
            "id": "cmpl-1",
            "object": "text_completion",
            "created": 0,
            "model": "m",
            "choices": [{
                "index": 0,
                "text": "",
                "finish_reason": "length",
                "logprobs": {
                    "tokens": ["token_id:7"],
                    "token_logprobs": [-0.1],
                    "top_logprobs": [top_logprobs],
                    "text_offset": [],
                },
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
            "nvext": nvext,
        }))
        .unwrap()
    }

    #[test]
    fn reads_label_logprobs_by_token_id() {
        let r = response(
            json!([
                {"token": "token_id:7", "logprob": -0.1, "bytes": null},
                {"token": "token_id:3", "logprob": -2.5, "bytes": null},
                {"token": "token_id:9", "logprob": -0.1, "bytes": null},
            ]),
            Some(json!({"worker_id": {"decode_worker_id": 42}})),
        );
        let scored = read_scores(&r, &[3, 7]).unwrap_or_else(|_| panic!("scores"));
        assert_eq!(scored.label_logprobs, vec![-2.5, -0.1]);
        assert_eq!(scored.prompt_tokens, 12);
        assert_eq!(scored.worker_id, Some(42));
    }

    #[test]
    fn missing_label_logprobs_fail_the_request() {
        let r = response(json!([{"token": "token_id:7", "logprob": -0.1}]), None);
        let Err((status, _)) = read_scores(&r, &[3, 7]) else {
            panic!("a missing label must fail");
        };
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    }
}
