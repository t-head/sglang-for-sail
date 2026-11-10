use axum::{
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use serde::Serialize;

/// Cap on a non-2xx upstream body forwarded to the client. A worker validation
/// error on a `Union` field emits one entry per branch, each echoing `input`,
/// so one bad value can balloon into tens of KB that just repeats the request
/// back. Arbitrary value; it holds the useful prefix of every observed case.
pub const MAX_UPSTREAM_ERROR_BODY_BYTES: usize = 2048;

/// Truncate an upstream error body to [`MAX_UPSTREAM_ERROR_BODY_BYTES`].
///
/// The result is deliberately not valid JSON when it truncates: the marker
/// tells the client the body was cut here, rather than leaving it to conclude
/// the worker emitted malformed JSON. Cuts on a UTF-8 boundary so the prefix
/// stays decodable.
pub fn truncate_error_body(bytes: Bytes) -> Bytes {
    if bytes.len() <= MAX_UPSTREAM_ERROR_BODY_BYTES {
        return bytes;
    }
    let total = bytes.len();
    let mut end = MAX_UPSTREAM_ERROR_BODY_BYTES;
    // Back off off a continuation byte (0b10xxxxxx) to the codepoint start.
    while end > 0 && (bytes[end] & 0xC0) == 0x80 {
        end -= 1;
    }
    let mut out = Vec::with_capacity(end + 48);
    out.extend_from_slice(&bytes[..end]);
    out.extend_from_slice(format!("... [truncated, {total} bytes total]").as_bytes());
    Bytes::from(out)
}

#[derive(Serialize)]
struct ErrorResponse<'a> {
    error: ErrorDetail<'a>,
}

#[derive(Serialize)]
struct ErrorDetail<'a> {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'a str,
    message: &'a str,
}

pub const HEADER_X_SMG_ERROR_CODE: &str = "X-SMG-Error-Code";

pub fn internal_error(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::INTERNAL_SERVER_ERROR, code, message)
}

pub fn bad_request(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::BAD_REQUEST, code, message)
}

pub fn not_found(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::NOT_FOUND, code, message)
}

pub fn service_unavailable(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::SERVICE_UNAVAILABLE, code, message)
}

pub fn failed_dependency(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::FAILED_DEPENDENCY, code, message)
}

pub fn not_implemented(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::NOT_IMPLEMENTED, code, message)
}

pub fn bad_gateway(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::BAD_GATEWAY, code, message)
}

pub fn method_not_allowed(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::METHOD_NOT_ALLOWED, code, message)
}

pub fn create_error(
    status: StatusCode,
    code: impl Into<String>,
    message: impl Into<String>,
) -> Response {
    let code_str = code.into();
    let message_str = message.into();

    let mut headers = HeaderMap::with_capacity(1);
    headers.insert(
        HEADER_X_SMG_ERROR_CODE,
        HeaderValue::from_str(&code_str).unwrap(),
    );

    (
        status,
        headers,
        Json(ErrorResponse {
            error: ErrorDetail {
                error_type: status_code_to_str(status),
                code: &code_str,
                message: &message_str,
            },
        }),
    )
        .into_response()
}

fn status_code_to_str(status_code: StatusCode) -> &'static str {
    status_code
        .canonical_reason()
        .unwrap_or("Unknown Status Code")
}

pub fn extract_error_code_from_response<B>(response: &Response<B>) -> &str {
    response
        .headers()
        .get(HEADER_X_SMG_ERROR_CODE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bodies up to the cap must round-trip byte-exact; callers rely on
    /// small worker error payloads passing through unchanged.
    #[test]
    fn error_body_up_to_cap_is_untouched() {
        for body in [
            Bytes::from(r#"{"error":{"message":"bad request"}}"#),
            Bytes::from(vec![b'x'; MAX_UPSTREAM_ERROR_BODY_BYTES]),
        ] {
            assert_eq!(truncate_error_body(body.clone()), body);
        }
    }

    #[test]
    fn oversized_error_body_is_truncated_with_marker() {
        let total = MAX_UPSTREAM_ERROR_BODY_BYTES * 4;
        let out = truncate_error_body(Bytes::from(vec![b'x'; total]));
        let text = String::from_utf8(out.to_vec()).expect("truncated body must stay UTF-8");
        assert!(text.starts_with("xxxx"), "prefix preserved: {text:.32}");
        assert!(
            text.ends_with(&format!("... [truncated, {total} bytes total]")),
            "marker must report the original size; got tail: {}",
            &text[text.len().saturating_sub(48)..],
        );
        assert!(out.len() < total, "must shrink: {} vs {total}", out.len());
    }

    /// A cut landing mid-codepoint must back off, or the client gets an
    /// undecodable tail.
    #[test]
    fn truncation_respects_utf8_boundaries() {
        // 3-byte chars do not divide evenly into the cap, so some cut lands
        // inside a codepoint regardless of alignment.
        let body: String = "错".repeat(MAX_UPSTREAM_ERROR_BODY_BYTES);
        let out = truncate_error_body(Bytes::from(body.into_bytes()));
        String::from_utf8(out.to_vec()).expect("truncated body must stay valid UTF-8");
    }
}
