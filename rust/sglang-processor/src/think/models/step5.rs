//! Step5Detector in the Python reasoning parser.

use crate::think::{ReasoningDefault, TagConfig};

pub(super) const THINK: TagConfig = TagConfig {
    start: "<think>",
    end: "</think>",
    tool_start: Some("<tool_call>"),
    reasoning_default: ReasoningDefault::on_unless("enable_thinking"),
    tool_start_at_line_start: false,
    force_reasoning_by_default: true,
    prefer_tool_start_boundary: false,
};
