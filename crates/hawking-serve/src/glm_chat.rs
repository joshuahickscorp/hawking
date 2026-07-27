//! GLM-5.2 chat-template rendering for the artifact's real `chat_template.jinja`.
//!
//! We do not run a general Jinja engine. The sealed Math-Preserve template is a
//! known, small dialect; this module implements the **non-tools text path** that
//! matches the artifact template byte-for-byte for ordinary system/user/assistant
//! turns. Tool-calling branches of the Jinja are intentionally refused rather
//! than guessed — a wrong tool render is worse than a clear error.

/// One chat message for GLM template application.
#[derive(Debug, Clone)]
pub struct GlmMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

/// Apply the artifact's GLM chat template to a simple (no-tools) conversation.
///
/// `template` must be the raw text of the artifact's `chat_template.jinja`. We
/// verify diagnostic markers from that file so a foreign/missing template fails
/// loudly instead of silently producing a wrong prompt.
///
/// `enable_thinking`: when true (GLM default), the generation prompt opens an
/// unclosed `<think>` so the model emits a reasoning block first. When false,
/// generation starts with an empty `<think></think>`.
pub fn render_glm_chat(
    template: &str,
    messages: &[GlmMessage<'_>],
    enable_thinking: bool,
) -> Result<String, String> {
    validate_glm_template(template)?;
    if messages.is_empty() {
        return Err("GLM chat template requires at least one message".into());
    }

    // Mirror the Jinja: default reasoning effort is "max" unless the caller
    // asked for high. Serve has no reasoning_effort knob yet, so "max".
    let mut out = String::from("[gMASK]<sop>");
    if enable_thinking {
        out.push_str("<|system|>Reasoning Effort: Max");
    }

    // last_user_index: used by the template to decide whether past assistant
    // turns keep their thinking content. We never carry reasoning_content on
    // the wire here, so past assistants always get an empty <think></think>.
    let last_user_index = messages
        .iter()
        .rposition(|m| m.role == "user")
        .unwrap_or(usize::MAX);

    for (i, m) in messages.iter().enumerate() {
        match m.role {
            "system" => {
                out.push_str("<|system|>");
                out.push_str(m.content);
            }
            "user" => {
                out.push_str("<|user|>");
                out.push_str(m.content);
            }
            "assistant" => {
                out.push_str("<|assistant|>");
                // Template: empty think for history before the last user turn
                // (clear_thinking default). We never inject reasoning_content.
                let _ = last_user_index; // reserved for future reasoning_content
                let _ = i;
                out.push_str("<think></think>");
                let trimmed = m.content.trim();
                if !trimmed.is_empty() {
                    out.push_str(trimmed);
                }
            }
            "tool" => {
                return Err(
                    "GLM chat template tool/observation turns are not rendered by the serve \
                     path yet; refuse rather than guess a tool render"
                        .into(),
                );
            }
            other => {
                return Err(format!(
                    "GLM chat template: unsupported role {other:?} (expected system/user/assistant)"
                ));
            }
        }
    }

    // add_generation_prompt = true (OpenAI chat completions always generate).
    out.push_str("<|assistant|>");
    if enable_thinking {
        out.push_str("<think>");
    } else {
        out.push_str("<think></think>");
    }
    Ok(out)
}

fn validate_glm_template(template: &str) -> Result<(), String> {
    // Markers that every real GLM-5.2 chat_template.jinja carries. Missing any
    // of them means this is not the artifact template — fail closed.
    for marker in [
        "[gMASK]<sop>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "add_generation_prompt",
        "<think>",
    ] {
        if !template.contains(marker) {
            return Err(format!(
                "chat template is not a GLM-5.2 template (missing marker {marker:?}); \
                 refusing to render"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_GLM_TEMPLATE: &str = r#"[gMASK]<sop>
{%- if add_generation_prompt -%}
    <|assistant|>{{- '<think></think>' if (enable_thinking is defined and not enable_thinking) else '<think>' -}}
{%- endif -%}
<|user|><|system|><think>
"#;

    #[test]
    fn renders_simple_user_turn_with_thinking() {
        let prompt = render_glm_chat(
            MINIMAL_GLM_TEMPLATE,
            &[GlmMessage {
                role: "user",
                content: "Hello",
            }],
            true,
        )
        .unwrap();
        assert!(prompt.starts_with("[gMASK]<sop><|system|>Reasoning Effort: Max"));
        assert!(prompt.contains("<|user|>Hello"));
        assert!(prompt.ends_with("<|assistant|><think>"));
    }

    #[test]
    fn renders_without_thinking() {
        let prompt = render_glm_chat(
            MINIMAL_GLM_TEMPLATE,
            &[GlmMessage {
                role: "user",
                content: "Hi",
            }],
            false,
        )
        .unwrap();
        assert!(prompt.starts_with("[gMASK]<sop><|user|>Hi"));
        assert!(prompt.ends_with("<|assistant|><think></think>"));
        assert!(!prompt.contains("Reasoning Effort"));
    }

    #[test]
    fn refuses_foreign_template() {
        let err = render_glm_chat("not a glm template", &[], true).unwrap_err();
        assert!(err.contains("missing marker"));
    }

    #[test]
    fn refuses_tool_role() {
        let err = render_glm_chat(
            MINIMAL_GLM_TEMPLATE,
            &[GlmMessage {
                role: "tool",
                content: "x",
            }],
            false,
        )
        .unwrap_err();
        assert!(err.contains("tool"));
    }
}
