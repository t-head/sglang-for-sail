// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache policy and usage survive the real router's body and transport paths.
//! Defaults belong to the worker: omitted options and `{}` remain unchanged.

use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::buckets_reorg::{Bucket, BucketGroups, BucketResolver, EngineGroup};
use sgl_router::config::{Cli, ParamSpec, SamplingField};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::policies_reorg::power_of_two::PowerOfTwoPolicy;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::{AppContext, ChatRouting};
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;
use crate::common::streaming::collect_body;

// This built-in encoder exercises actual router-generated input_ids, while
// the tiny fixture keeps tokenization local and independent of model downloads.
const MODEL: &str = "deepseek-v4-tiny";
const AUTHORIZATION: &str = "Bearer cache-tenant-key";
const SSE_PARTS: [&str; 4] = [
    "data: {\"id\":\"cache-decode\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
    "data: {\"id\":\"cache-decode\",\"choices\":[],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":100,\"total_tokens\":1100,\"prompt_tokens_details\":{\"cached_tokens\":200,\"cache_write_",
    "tokens\":700}}}\n\n",
    "data: [DONE]\n\n",
];
const SSE_WITHOUT_USAGE: [&str; 2] = [SSE_PARTS[0], SSE_PARTS[3]];

fn response_headers(writes: [&str; 2], mixed_case: bool) -> HeaderMap {
    let names = if mixed_case {
        [
            "Msh-Usage-Cache-Write-Tokens-5m",
            "MSH-USAGE-CACHE-WRITE-TOKENS-1H",
        ]
    } else {
        [
            "msh-usage-cache-write-tokens-5m",
            "msh-usage-cache-write-tokens-1h",
        ]
    };
    let mut headers = HeaderMap::new();
    for (name, value) in names.into_iter().zip(writes) {
        headers.insert(
            HeaderName::from_bytes(name.as_bytes()).unwrap(),
            HeaderValue::from_str(value).unwrap(),
        );
    }
    for name in [
        "set-cookie",
        "openai-organization",
        "openai-project",
        "authorization",
        "msh-usage-cache-write-tokens-30m",
    ] {
        headers.insert(
            HeaderName::from_static(name),
            HeaderValue::from_static("private-upstream-value"),
        );
    }
    headers
}

fn options() -> Vec<Option<Value>> {
    vec![
        None,
        Some(json!({})),
        Some(json!({"mode": "implicit"})),
        Some(json!({"mode": "implicit", "ttl": "5m"})),
        Some(json!({"mode": "implicit", "ttl": "1h"})),
    ]
}

fn response_body() -> Value {
    json!({
        "id": "cache-decode",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "prompt_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 700}
        }
    })
}

fn context(
    prefill: &MockWorker,
    decode: Option<&MockWorker>,
    reorg: bool,
    forward_ids: bool,
    sampling_defaults: bool,
) -> Arc<AppContext> {
    let mut cfg = <Cli as clap::Parser>::parse_from([
        "sgl-router",
        "--model-id",
        MODEL,
        "--tokenizer-path",
        "tests/fixtures/tiny_tokenizer.json",
        "--worker-urls",
        "http://placeholder:0",
        "--policy",
        "round_robin",
    ])
    .into_config()
    .unwrap();
    cfg.model.disable_input_ids_forwarding = !forward_ids;
    if sampling_defaults {
        cfg.model.sampling_overrides.params.insert(
            SamplingField::Temperature,
            ParamSpec::Exact(serde_json::Number::from(1)),
        );
    }
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    if forward_ids {
        assert!(tokenizers.has_chat_formatter(MODEL));
    }
    let registry = Arc::new(WorkerRegistry::default());
    registry
        .add(WorkerSpec {
            id: WorkerId("prefill-or-plain".into()),
            url: prefill.url.clone(),
            mode: if decode.is_some() {
                WorkerMode::Prefill
            } else {
                WorkerMode::Plain
            },
            model_ids: vec![ModelId(MODEL.into())],
            bootstrap_port: Some(8997),
            ..Default::default()
        })
        .unwrap();
    if let Some(decode) = decode {
        registry
            .add(WorkerSpec {
                id: WorkerId("decode".into()),
                url: decode.url.clone(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId(MODEL.into())],
                bootstrap_port: None,
                ..Default::default()
            })
            .unwrap();
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let mut ctx = AppContext::new(
        cfg,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        registry,
        policies,
    );
    if reorg {
        let group = || {
            EngineGroup::new(Arc::new(PowerOfTwoPolicy::new(Arc::clone(
                &ctx.engine_reported_load,
            ))))
        };
        let groups = if decode.is_some() {
            BucketGroups::Pd {
                prefill: group(),
                decode: group(),
            }
        } else {
            BucketGroups::Plain(group())
        };
        ctx.chat_routing = ChatRouting::Reorg(
            [(
                ModelId(MODEL.into()),
                BucketResolver::new(vec![Bucket::new("default", groups)]).unwrap(),
            )]
            .into_iter()
            .collect(),
        );
    }
    Arc::new(ctx)
}

fn request_body(options: Option<&Value>, streaming: bool) -> Value {
    let mut body = json!({
        "model": MODEL,
        "cache_user_id": "gateway-user-42",
        "messages": [{"role": "user", "content": "hi"}],
        "rid": "client-cache-request",
        "stream": streaming,
        "max_tokens": 100
    });
    if let Some(options) = options {
        body["prompt_cache_options"] = options.clone();
    }
    if streaming {
        body["stream_options"] = json!({"include_usage": true});
    }
    body
}

async fn send(ctx: Arc<AppContext>, body: &Value, writes: [&str; 2]) -> bytes::Bytes {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", AUTHORIZATION)
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap();
    let response = build_router(ctx).oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK, "body={body}");
    // Inspect initial headers before collecting any JSON or SSE body bytes.
    for (name, expected) in [
        ("Msh-Usage-Cache-Write-Tokens-5m", writes[0]),
        ("MSH-USAGE-CACHE-WRITE-TOKENS-1H", writes[1]),
    ] {
        assert_eq!(
            response.headers().get(name).and_then(|v| v.to_str().ok()),
            Some(expected),
            "initial response header {name}",
        );
    }
    for name in [
        "set-cookie",
        "openai-organization",
        "openai-project",
        "authorization",
        "msh-usage-cache-write-tokens-30m",
    ] {
        assert!(response.headers().get(name).is_none(), "must strip {name}");
    }
    collect_body(response.into_body()).await
}

fn captured(worker: &MockWorker, original: &Value, forward_ids: bool) -> Value {
    let capture = worker.captured.lock().unwrap();
    assert_eq!(
        capture.headers.get("authorization").map(String::as_str),
        Some(AUTHORIZATION),
    );
    let forwarded: Value = serde_json::from_slice(
        capture
            .last_body
            .as_ref()
            .expect("worker received the request"),
    )
    .unwrap();
    assert_eq!(
        forwarded.get("prompt_cache_options"),
        original.get("prompt_cache_options"),
    );
    assert_eq!(forwarded.get("cache_user_id"), original.get("cache_user_id"));
    assert_eq!(forwarded["messages"], original["messages"]);
    if forward_ids {
        assert!(!forwarded["input_ids"].as_array().unwrap().is_empty());
    } else {
        assert!(forwarded.get("input_ids").is_none());
    }
    forwarded
}

#[tokio::test]
async fn plain_options_survive_sampling_splice_and_input_ids_rebuild() {
    let expected = response_body();
    let worker = MockWorker::start_json_with_response_headers(
        expected.clone(),
        response_headers(["700", "0"], true),
    )
    .await;
    for reorg in [false, true] {
        for forward_ids in [false, true] {
            let ctx = context(&worker, None, reorg, forward_ids, true);
            for options in options() {
                let body = request_body(options.as_ref(), false);
                let response = send(Arc::clone(&ctx), &body, ["700", "0"]).await;
                assert_eq!(response.as_ref(), expected.to_string().as_bytes());
                let forwarded = captured(&worker, &body, forward_ids);
                assert_eq!(forwarded["temperature"], 1);
                assert_eq!(forwarded["rid"], body["rid"]);
                assert!(forwarded.get("bootstrap_room").is_none());
            }
        }
    }
}

#[tokio::test]
async fn unmodified_plain_body_and_stream_usage_pass_through() {
    let worker = MockWorker::start_with_response_headers(
        SSE_PARTS.to_vec(),
        response_headers(["0", "700"], false),
    )
    .await;
    for reorg in [false, true] {
        let ctx = context(&worker, None, reorg, false, false);
        for options in options() {
            let body = request_body(options.as_ref(), true);
            let response = send(Arc::clone(&ctx), &body, ["0", "700"]).await;
            assert_eq!(response.as_ref(), SSE_PARTS.concat().as_bytes());
            assert_eq!(captured(&worker, &body, false), body);
            let captured_body = worker.captured.lock().unwrap().last_body.clone().unwrap();
            assert_eq!(
                captured_body.as_ref(),
                serde_json::to_vec(&body).unwrap().as_slice(),
            );
        }
    }
}

#[tokio::test]
async fn pd_preserves_options_and_returns_only_decode_json_or_stream_usage() {
    // Prefill's conflicting counters must never replace or add to decode usage.
    let prefill = MockWorker::start_json_with_response_headers(
        json!({"usage": {"prompt_tokens_details": {
            "cached_tokens": 9999, "cache_write_tokens": 9999
        }}}),
        response_headers(["9999", "8888"], true),
    )
    .await;
    for streaming in [false, true] {
        let expected = response_body();
        let writes = if streaming {
            ["700", "0"]
        } else {
            ["0", "700"]
        };
        let decode = if streaming {
            MockWorker::start_with_response_headers(
                SSE_PARTS.to_vec(),
                response_headers(writes, true),
            )
            .await
        } else {
            MockWorker::start_json_with_response_headers(
                expected.clone(),
                response_headers(writes, false),
            )
            .await
        };
        for reorg in [false, true] {
            for forward_ids in [false, true] {
                let ctx = context(&prefill, Some(&decode), reorg, forward_ids, true);
                for options in options() {
                    let body = request_body(options.as_ref(), streaming);
                    let response = send(Arc::clone(&ctx), &body, writes).await;
                    if streaming {
                        assert_eq!(response.as_ref(), SSE_PARTS.concat().as_bytes());
                    } else {
                        assert_eq!(response.as_ref(), expected.to_string().as_bytes());
                    }
                    let p = captured(&prefill, &body, forward_ids);
                    let d = captured(&decode, &body, forward_ids);
                    assert_eq!(p["stream"], false);
                    assert!(p.get("stream_options").is_none());
                    assert_eq!(d["stream"], streaming);
                    assert_eq!(d.get("stream_options"), body.get("stream_options"));
                    assert_eq!(p["temperature"], 1);
                    assert_eq!(d["temperature"], 1);
                    assert_eq!(p["input_ids"], d["input_ids"]);
                    assert_eq!(p["rid"], d["rid"]);
                    assert!(p["rid"].as_str().unwrap().starts_with("sgl-router-"));
                    assert_eq!(p["bootstrap_room"], d["bootstrap_room"]);
                    assert!(p["bootstrap_room"].as_u64().is_some());
                    assert_eq!(p["bootstrap_host"], "127.0.0.1");
                    assert_eq!(p["bootstrap_port"], 8997);
                }
            }
        }
    }
}

#[tokio::test]
async fn write_headers_survive_without_stream_usage_and_ignore_prefill_headers() {
    let prefill = MockWorker::start_json_with_response_headers(
        json!({"prefill": true}),
        response_headers(["9999", "8888"], true),
    )
    .await;
    for writes in [["0", "0"], ["700", "0"], ["0", "700"]] {
        for streaming in [false, true] {
            let mut expected = response_body();
            let total_write =
                writes[0].parse::<u64>().unwrap() + writes[1].parse::<u64>().unwrap();
            expected["usage"]["prompt_tokens_details"]["cache_write_tokens"] = json!(total_write);
            let worker = if streaming {
                MockWorker::start_with_response_headers(
                    SSE_WITHOUT_USAGE.to_vec(),
                    response_headers(writes, true),
                )
                .await
            } else {
                MockWorker::start_json_with_response_headers(
                    expected.clone(),
                    response_headers(writes, false),
                )
                .await
            };
            for reorg in [false, true] {
                for pd in [false, true] {
                    let ctx = if pd {
                        context(&prefill, Some(&worker), reorg, false, false)
                    } else {
                        context(&worker, None, reorg, false, false)
                    };
                    let mut body = request_body(None, streaming);
                    body.as_object_mut().unwrap().remove("stream_options");
                    let response = send(ctx, &body, writes).await;
                    if streaming {
                        assert_eq!(response.as_ref(), SSE_WITHOUT_USAGE.concat().as_bytes());
                    } else {
                        assert_eq!(response.as_ref(), expected.to_string().as_bytes());
                    }
                }
            }
        }
    }
}
