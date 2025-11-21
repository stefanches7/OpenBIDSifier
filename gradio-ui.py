#!/usr/bin/env python
"""
Gradio demo UI for the BIDSifierAgent.

This wraps the existing CLI-style step-wise logic (prompts.py + agent.py)
into an interactive Gradio interface.

Requirements
------------
    pip install gradio bids_validator python-dotenv dspy-ai
"""

from __future__ import annotations

import time
from bids_validator import BIDSValidator
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from agent import BIDSifierAgent  # your existing agent
from cli import parse_commands_from_markdown  # reuse the CLI helper if available

# Step mapping: UI label -> agent step id

BIDSIFIER_STEPS: Dict[str, str] = {
    "1. Summarize dataset": "summary",
    "2. Propose metadata commands": "create_metadata",
    "3. Propose structure commands": "create_structure",
    "4. Propose rename/move commands": "rename_move",
}

STEP_LABELS = list(BIDSIFIER_STEPS.keys())
NUM_STEPS = len(STEP_LABELS)


# Helpers

def split_shell_commands(text: str) -> List[str]:
    """
    Split a multi-line shell script into individual commands.

    Each non-empty line is treated as a separate command, except when a line
    ends with a backslash (\\), in which case it is joined with the following
    line(s) to form a single logical command.

    Parameters
    ----------
    text : str
        Multi-line string containing shell commands.

    Returns
    -------
    list of str
        The list of shell commands to execute.
    """
    commands: List[str] = []
    current: str = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if current:
            # Continue an ongoing command
            if line.endswith("\\"):
                current += " " + line[:-1].rstrip()
            else:
                current += " " + line
                commands.append(current)
                current = ""
        else:
            # Start a new command
            if line.endswith("\\"):
                current = line[:-1].rstrip()
            else:
                commands.append(line)

    if current:
        commands.append(current)

    return commands


def build_context(
    dataset_xml: str,
    readme_text: str,
    publication_text: str,
    output_root: str,
) -> Dict[str, Any]:
    """
    Build the context dictionary expected by BIDSifierAgent.

    Parameters
    ----------
    dataset_xml : str
        Dataset XML content (or empty string).
    readme_text : str
        README text content (or empty string).
    publication_text : str
        Publication/notes content (or empty string).
    output_root : str
        Target BIDS root directory.

    Returns
    -------
    dict
        Context dictionary.
    """
    return {
        "dataset_xml": dataset_xml or None,
        "readme_text": readme_text or None,
        "publication_text": publication_text or None,
        "output_root": output_root or "./bids_output",
        "user_feedback": "",
    }


# Core callbacks

def call_bidsifier_step(
    openai_api_key: str,
    dataset_xml: str,
    readme_text: str,
    publication_text: str,
    output_root: str,
    provider: str,
    model: str,
    step_label: str,
    manual_prompt: str,
) -> Tuple[str, str, Dict[str, Any], int]:
    """
    Call BIDSifierAgent for a given step and return raw output + parsed commands.

    Parameters
    ----------
    dataset_xml : str
        Dataset XML content.
    readme_text : str
        README content.
    publication_text : str
        Publication/notes content.
    output_root : str
        Target BIDS root directory.
    provider : str
        LLM provider (e.g. "openai").
    model : str
        LLM model name (e.g. "gpt-5" or "gpt-4o-mini").
    step_label : str
        UI label of the selected step.
    manual_prompt : str
        Optional free-form user override; if non-empty we call `run_query`
        instead of the structured `run_step`.

    Returns
    -------
    llm_output : str
        Raw text returned by the LLM.
    commands_str : str
        Commands extracted from the first fenced bash/sh code block.
    state : dict
        State capturing last call inputs, for potential reuse (e.g. retry).
    step_index : int
        Index of the current step (for progress updates).
    """
    if not output_root.strip():
        return (
            "⚠️ Please provide an output root before calling BIDSifier.",
            "",
            {},
            0,
        )

    if step_label not in BIDSIFIER_STEPS:
        return (
            "⚠️ Please select a valid BIDSifier step.",
            "",
            {},
            0,
        )

    step_id = BIDSIFIER_STEPS[step_label]
    context = build_context(dataset_xml, readme_text, publication_text, output_root)

    agent = BIDSifierAgent(provider=provider, model=model, openai_api_key=openai_api_key)

    # Decide whether to use the structured step prompt or a free-form query:
    if manual_prompt.strip():
        llm_output = agent.run_query(manual_prompt)
    else:
        llm_output = agent.run_step(step_id, context)

    # Extract bash commands from fenced block
    commands = parse_commands_from_markdown(llm_output)
    commands_str = "\n".join(commands) if commands else ""

    # Step index for progress bar
    try:
        step_index = STEP_LABELS.index(step_label) + 1
    except ValueError:
        step_index = 0

    state = {
        "openai_api_key": openai_api_key,
        "dataset_xml": dataset_xml,
        "readme_text": readme_text,
        "publication_text": publication_text,
        "output_root": output_root,
        "provider": provider,
        "model": model,
        "step_label": step_label,
        "step_id": step_id,
        "llm_output": llm_output,
        "commands": commands,
    }

    return llm_output, commands_str, state, step_index


def confirm_commands(
    last_state: Optional[Dict[str, Any]],
    progress_value: int,
) -> Tuple[str, int]:
    """
    Confirm and execute the commands proposed in the last LLM response.

    Parameters
    ----------
    last_state : dict or None
        State returned by `call_bidsifier_step`, containing `output_root` and
        list of `commands`. If None, nothing can be executed.
    progress_value : int
        Current progress through steps.

    Returns
    -------
    status_message : str
        Combined stdout/stderr/exit codes of executed commands.
    new_progress : int
        Updated progress value.
    """
    if not last_state:
        return "⚠️ No previous BIDSifier step to confirm.", progress_value

    output_root = last_state.get("output_root", "").strip()
    commands: List[str] = last_state.get("commands", [])

    if not output_root:
        return "⚠️ Output root is empty; cannot execute commands.", progress_value

    if not commands:
        return "⚠️ No commands detected in the last BIDSifier output.", progress_value

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    all_details: List[str] = []

    for raw_cmd in commands:
        # Support multi-line shell with backslash continuations *within* a line,
        # but commands already come one-per-line from `parse_commands_from_markdown`.
        for cmd in split_shell_commands(raw_cmd):
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(root),
                capture_output=True,
                text=True,
            )
            all_details.append(
                f"Executed: {cmd}\n"
                f"Exit code: {proc.returncode}\n"
                f"Stdout:\n{proc.stdout}\n"
                f"Stderr:\n{proc.stderr}\n" + "-" * 40
            )

    status = "### Command execution log\n\n" + "\n\n".join(all_details)

    # Heuristic: if we executed commands for this step, bump progress to max
    # of current and the step index.
    step_label = last_state.get("step_label")
    try:
        idx = STEP_LABELS.index(step_label)
        new_progress = max(progress_value, idx + 1)
    except (ValueError, TypeError):
        new_progress = progress_value

    return status, new_progress


def run_bids_validation(output_root: str) -> Tuple[str, str]:
    """
    Run the BIDS filename validator on all files under `output_root`.

    Parameters
    ----------
    output_root : str
        Root directory of the BIDS dataset.

    Returns
    -------
    report : str
        A Markdown report summarizing which files are BIDS-like and which are not.
    status_token : str
        "pass:<timestamp>" if all files are BIDS-compliant (at least one file),
        otherwise "fail:<timestamp>". The timestamp ensures Gradio's .change
        event fires every time.
    """
    if not output_root.strip():
        return (
            "⚠️ Please provide an output root before running the BIDS validator.",
            f"fail:{time.time()}",
        )

    root = Path(output_root)
    if not root.exists():
        return (
            f"⚠️ Output root `{output_root}` does not exist. Nothing to validate.",
            f"fail:{time.time()}",
        )

    validator = BIDSValidator()

    lines = []
    valid_count = 0
    invalid_count = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        rel_str = "/" + rel.as_posix()
        is_valid = validator.is_bids(rel_str)
        if is_valid:
            valid_count += 1
            status = "OK"
        else:
            invalid_count += 1
            status = "NOT BIDS"
        lines.append(f"{rel_str}: {status}")

    if not lines:
        return (
            f"Note: No files found under `{output_root}` to validate.",
            f"fail:{time.time()}",
        )

    summary = (
        f"Validated {valid_count + invalid_count} files: "
        f"{valid_count} OK, {invalid_count} NOT BIDS."
    )
    bullet_lines = "\n".join(f"- `{line}`" for line in lines)

    report = f"### BIDS Validator report\n\n{bullet_lines}\n\n**Summary:** {summary}"

    status_flag = "pass" if invalid_count == 0 and valid_count > 0 else "fail"
    status_token = f"{status_flag}:{time.time()}"
    return report, status_token


# Gradio UI

with gr.Blocks(
    title="BIDSifier Agent Interface",
    theme=gr.themes.Citrus(),
    head="""
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.2/dist/confetti.browser.min.js"></script>
    """,
) as demo:
    gr.Image(
        value="images/bh_logo.png",
        show_label=False,
        height=80,
        elem_id="bh_logo",
    )

    gr.Markdown(
        """
        # BIDSifier Agent Demo

        Interactive UI wrapping the **BIDSifierAgent** (CLI logic) to propose
        shell commands for BIDS conversion, step by step.
        Commands are extracted from fenced ```bash```/```sh``` blocks.
        """
    )

    with gr.Row():
        openai_key_input = gr.Textbox(
            label="OpenAI API Key",
            placeholder="Paste your OpenAI API key here",
            lines=1,
            type="password",
        )
        dataset_xml_input = gr.Textbox(
            label="Dataset XML",
            placeholder="Paste dataset_structure.xml content here (optional)",
            lines=8,
        )
        readme_input = gr.Textbox(
            label="README",
            placeholder="Paste README.md content here (optional)",
            lines=8,
        )

    publication_input = gr.Textbox(
        label="Publication / Notes",
        placeholder="Paste relevant publication snippets or notes here (optional)",
        lines=6,
    )

    with gr.Accordion("LLM settings (advanced)", open=False):
        provider_input = gr.Dropdown(
            label="Provider",
            choices=["openai"],
            value="openai",
        )
        model_input = gr.Textbox(
            label="Model",
            value="gpt-4o-mini",
            placeholder="e.g., gpt-4o-mini, gpt-5",
        )

    output_root_input = gr.Textbox(
        label="Output root",
        placeholder="brainmets-bids",
        lines=1,
    )

    step_dropdown = gr.Dropdown(
        label="BIDSifier step",
        choices=STEP_LABELS,
        value=STEP_LABELS[0],
        info="Select the current logical step in the BIDSifier workflow.",
    )

    progress_bar = gr.Slider(
        label="Progress through BIDSifier steps",
        minimum=0,
        maximum=NUM_STEPS,
        step=1,
        value=0,
        interactive=False,
    )

    manual_prompt_input = gr.Textbox(
        label="Override prompt / free-form query (optional)",
        placeholder=(
            "If non-empty, this free-form query will be sent to the agent instead "
            "of the structured step prompt."
        ),
        lines=3,
    )

    call_button = gr.Button("Call BIDSifier", variant="primary")

    llm_output_box = gr.Textbox(
        label="Raw BIDSifier output",
        lines=10,
        interactive=True,
    )

    commands_box = gr.Textbox(
        label="Parsed shell commands (from fenced bash block)",
        lines=10,
        interactive=True,
    )

    confirm_button = gr.Button("Confirm / Run commands", variant="primary")
    bids_validator_button = gr.Button("Run BIDS Validator", variant="primary")

    status_msg = gr.Markdown(label="Status / execution log")
    validation_status = gr.Textbox(visible=False)

    # State to store last agent call for Confirm
    last_state = gr.State(value=None)

    # Wiring

    call_button.click(
        fn=call_bidsifier_step,
        inputs=[
            openai_key_input,
            dataset_xml_input,
            readme_input,
            publication_input,
            output_root_input,
            provider_input,
            model_input,
            step_dropdown,
            manual_prompt_input,
        ],
        outputs=[llm_output_box, commands_box, last_state, progress_bar],
    )

    confirm_button.click(
        fn=confirm_commands,
        inputs=[last_state, progress_bar],
        outputs=[status_msg, progress_bar],
    )

    bids_validator_button.click(
        fn=run_bids_validation,
        inputs=[output_root_input],
        outputs=[status_msg, validation_status],
    )

    validation_status.change(
        fn=None,
        inputs=[validation_status],
        outputs=[],
        js="""
        (value) => {
            if (value && value.startsWith("pass") && window.confetti) {
                window.confetti({
                    particleCount: 240,
                    spread: 70,
                    origin: { y: 0.6 }
                });
            }
            return [];
        }
        """,
    )


if __name__ == "__main__":
    demo.launch()
