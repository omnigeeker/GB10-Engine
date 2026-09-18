//! Chat-template rendering.
//!
//! The checkpoint ships a Jinja `chat_template.jinja`; we render *that file*
//! with `minijinja` rather than reimplementing the format, so prompt bytes stay
//! identical to the HF reference. Only `raise_exception` needs supplying, as it
//! is a HuggingFace extension rather than a Jinja builtin.

use minijinja::{Environment, Value};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// One message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
}

/// Options mirroring the variables the Qwen3.5 template reads.
#[derive(Debug, Clone)]
pub struct ChatTemplateOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    /// `xhigh` (default) | `medium` | `low`
    pub reasoning_effort: String,
    pub preserve_thinking: bool,
    pub add_vision_id: bool,
}

impl Default for ChatTemplateOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: true,
            reasoning_effort: "xhigh".to_string(),
            preserve_thinking: true,
            add_vision_id: false,
        }
    }
}

pub struct ChatTemplate {
    env: Environment<'static>,
    source: String,
}

impl ChatTemplate {
    /// Load `chat_template.jinja` from a checkpoint directory, falling back to
    /// the `tokenizer_config.json` `chat_template` field.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        let jinja = dir.join("chat_template.jinja");
        if jinja.exists() {
            return Self::from_template(&std::fs::read_to_string(jinja)?);
        }
        let tc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json"))?)?;
        let tpl = tc
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("no chat_template.jinja and no tokenizer_config chat_template"))?;
        Self::from_template(tpl)
    }

    pub fn from_template(source: &str) -> anyhow::Result<Self> {
        let mut env = Environment::new();
        env.add_function("raise_exception", |msg: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        });
        // HF Jinja exposes `str.startswith` / `str.endswith` as methods.
        // minijinja has neither, so supply them here rather than editing the
        // checkpoint's template — prompt bytes must stay faithful.
        env.set_unknown_method_callback(|_state, value, method, args| {
            if let Some(s) = value.as_str() {
                let arg = args.first().and_then(|v| v.as_str()).unwrap_or("");
                match method {
                    "startswith" => return Ok(Value::from(s.starts_with(arg))),
                    "endswith" => return Ok(Value::from(s.ends_with(arg))),
                    _ => {}
                }
            }
            Err(minijinja::Error::new(
                minijinja::ErrorKind::UnknownMethod,
                format!("unknown method {method}"),
            ))
        });
        // `tojson` exists in minijinja; HF templates also use `|string`,
        // `|trim`, `|items`, `|safe` and `|default`, all built in.
        // `add_template_owned` keeps the source inside the environment, which
        // is what lets us hold an `Environment<'static>`.
        env.add_template_owned("chat", source.to_string())
            .map_err(|e| anyhow::anyhow!("compiling chat template: {e}"))?;
        Ok(Self {
            env,
            source: source.to_string(),
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Render a conversation to the exact prompt string fed to the tokenizer.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        opts: &ChatTemplateOptions,
    ) -> anyhow::Result<String> {
        let tpl = self.env.get_template("chat")?;
        let msgs = Value::from_serialize(messages);
        let out = tpl
            .render(minijinja::context! {
                messages => msgs,
                add_generation_prompt => opts.add_generation_prompt,
                enable_thinking => opts.enable_thinking,
                reasoning_effort => opts.reasoning_effort.clone(),
                preserve_thinking => opts.preserve_thinking,
                add_vision_id => opts.add_vision_id,
                tools => Value::from(()),
            })
            .map_err(|e| anyhow::anyhow!("rendering chat template: {e}"))?;
        Ok(out)
    }
}

/// Convenience: build a plain text-only message.
pub fn text_message(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content: Value::from(content),
        reasoning_content: None,
        tool_calls: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::QwenTokenizer;

    fn model_dir() -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from("../../models/Qwen3.8-27B-NVFP4");
        p.exists().then_some(p)
    }

    /// Fixture copied from the HF oracle: rendered prompt + exact token ids.
    fn fixtures() -> serde_json::Value {
        serde_json::from_str(include_str!("../tests/fixtures/chat_parity.json")).unwrap()
    }

    #[test]
    fn renders_prompt_bytes_identical_to_hf_oracle() {
        let Some(dir) = model_dir() else {
            eprintln!("skipping: model not present");
            return;
        };
        let tpl = ChatTemplate::from_model_dir(&dir).unwrap();
        let fx = fixtures();
        for case in fx["cases"].as_array().unwrap() {
            let user = case["prompt"].as_str().unwrap();
            let want = case["rendered_prompt"].as_str().unwrap();
            let msgs = vec![text_message("user", user)];
            let got = tpl.render(&msgs, &ChatTemplateOptions::default()).unwrap();
            assert_eq!(got, want, "rendered prompt mismatch for {user:?}");
        }
    }

    #[test]
    fn tokenizes_prompts_identical_to_hf_oracle() {
        let Some(dir) = model_dir() else { return };
        let tpl = ChatTemplate::from_model_dir(&dir).unwrap();
        let tok = QwenTokenizer::from_model_dir(&dir).unwrap();
        let fx = fixtures();
        for case in fx["cases"].as_array().unwrap() {
            let user = case["prompt"].as_str().unwrap();
            let want: Vec<u32> = case["prompt_token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let rendered = tpl
                .render(&[text_message("user", user)], &ChatTemplateOptions::default())
                .unwrap();
            let got = tok.encode(&rendered, false).unwrap();
            assert_eq!(got, want, "token ids mismatch for {user:?}");
        }
    }

    #[test]
    fn rejects_empty_conversation() {
        let Some(dir) = model_dir() else { return };
        let tpl = ChatTemplate::from_model_dir(&dir).unwrap();
        let err = tpl.render(&[], &ChatTemplateOptions::default());
        assert!(err.is_err(), "template must reject an empty message list");
    }
}
