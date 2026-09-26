#!/usr/bin/env python3
"""The MMLU prompt, defined once.

Three implementations score these questions -- the Rust engine, the
transformers reference, and llama.cpp. If any two build a different string the
comparison measures the prompt instead of the weights, so all three use this
rule.

It must match `render_choice_prompt()` in crates/gb10-verify/src/main.rs
exactly. The engine records its token count per question and the reference
scorer compares, which is what turns "the strings match" from a claim into a
checked fact.
"""


def render(q):
    s = ("The following are multiple choice questions (with answers) about "
         f"{q['subject'].replace('_', ' ')}.\n\n{q['question'].strip()}\n")
    for i, o in enumerate(q["options"]):
        s += f"{chr(65 + i)}. {o.strip()}\n"
    return s + "Answer:"
