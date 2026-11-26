//! Step5Detector in the Python reasoning parser.

use crate::think::ThinkConfig;

pub(super) const THINK: ThinkConfig = ThinkConfig {
    start: "<think>",
    end: "</think>",
    tool_start: Some("<tool_call>"),
    tool_start_at_line_start: false,
    force_reasoning_by_default: true,
    prefer_tool_start_boundary: true,
};
