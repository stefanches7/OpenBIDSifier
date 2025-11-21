#!/usr/bin/env python
"""
Demo Gradio interface for a BIDS conversion assistant.

Requirements
------------
    pip install gradio bids_validator
"""

from __future__ import annotations

import time
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
from bids_validator import BIDSValidator

BIDS_STEPS: Dict[str, Dict[str, str]] = {
    "1.1 Modality agnostic files: dataset_description.json": {
        "output_file": "dataset_description.json",
        "default_prompt": (
            "You are assisting with converting a neuroimaging dataset to BIDS.\n"
            "Current step: dataset_description.json (Dataset description).\n\n"
            "Dataset XML: {dataset_xml}\n"
            "Output root (BIDS root): {output_root}\n\n"
            "Task:\n"
            "- Inspect the dataset structure (as described in the XML).\n"
            "- Propose modality-agnostic BIDS files that should be created or updated.\n"
            "- Highlight any potential issues or missing metadata.\n"
        ),
    },
    "1.2. Modality agnostic files: README.md": {
        "output_file": "README.md",
        "default_prompt": ("..."),
    },
    "1.3. Modality agnostic files: LICENSE (optional)": {
        "output_file": "LICENSE",
        "default_prompt": ("..."),
    },
    "1.4. Modality agnostic files: participants.tsv (recommended)": {
        "output_file": "participants.tsv",
        "default_prompt": ("..."),
    },
    "1.5. Modality agnostic files: participants.json (recommended)": {
        "output_file": "participants.json",
        "default_prompt": ("..."),
    },
    "2. Modality-specific files: MRI, MEG, EEG, ...": {
        # No output_file here: we expect a script/command for this step.
        "default_prompt": ("..."),
    },
    # We moved this to an extra button in bottom
    # "3. Validation: BIDS validation": {
    #     "default_prompt": ("..."),
    # },
}

# For ordered access to steps and progress handling.
STEP_KEYS = list(BIDS_STEPS.keys())
NUM_STEPS = len(STEP_KEYS)

# Load canvas-confetti JS code so it can be inlined into the HTML head.
# CONFETTI_JS = Path("js/confetti.browser.min.js").read_text(encoding="utf-8")


def load_css() -> str:
    """
    Load custom CSS from a file named `style.css`.

    Returns
    -------
    str
        The CSS content as a string.
    """
    with open("style.css", "r", encoding="utf-8") as file:
        css_content = file.read()
    return css_content


def get_default_prompt(step: str, dataset_xml: str, output_root: str) -> str:
    """
    Generate the default prompt template for a given BIDS step.

    Parameters
    ----------
    step : str
        Name of the selected BIDS conversion step.
    dataset_xml : str
        Path or content of the dataset XML file.
    output_root : str
        Path to the BIDS output root.

    Returns
    -------
    str
        The default prompt text for the selected step. If the step is unknown,
        an empty string is returned.
    """
    config = BIDS_STEPS.get(step)
    if config is None:
        return ""

    template = config.get("default_prompt", "")
    return template.format(
        dataset_xml=dataset_xml or "<dataset_xml_not_set>",
        output_root=output_root or "<output_root_not_set>",
    )


def generate_response(
    dataset_xml: str,
    publication: str,
    readme: str,
    extra_info: str,
    output_root: str,
    provider: str,
    model: str,
    step: str,
    prompt: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Simulate an LLM response for demonstration purposes.

    This function is where you would call your actual LLM in a real system.

    Parameters
    ----------
    dataset_xml : str
        Path or content of the dataset XML file.
    publication : str
        Path or content of the publication snippets file.
    readme : str
        Path or content of the README file.
    extra_info : str
        Extra information to give to the LLM.
    output_root : str
        Path to the BIDS output root directory.
    provider : str
        LLM provider (e.g., "openai").
    model : str
        LLM model name (e.g., "gpt-5").
    step : str
        Name of the selected BIDS conversion step.
    prompt : str
        Prompt text that will be sent to the LLM.

    Returns
    -------
    response : str
        Dummy LLM output for the demo.
    state : dict
        A dictionary capturing the inputs used for this generation, so that
        they can be reused by the retry callback.
    """
    if not step:
        response = "⚠️ Please select a BIDS step before generating a response."
    elif not dataset_xml or not output_root:
        response = (
            "⚠️ Please provide at least `dataset_xml` and `output_root` "
            "before generating a response."
        )
    else:
        # TODO: Replace this with an actual LLM call
        response = """
{
    "Name": "Stanford Brain Metastases MRI (multisequence) \u2014 preliminary BIDS scaffold",
    "BIDSVersion": "1.9.0",
    "DatasetType": "raw",
    "License": "TODO: Add license (check original dataset terms; e.g., CC BY-NC 4.0 or institutional terms)",
    "Acknowledgements": "Images reported as skull-stripped in source; verify BIDS 'raw' compliance or consider labeling as derivatives if needed.",
    "Authors": [
        "Darvin Yi",
        "Endre Gr\u00f8vik",
        "Elizabeth Tong",
        "Michael Iv",
        "Daniel Rubin",
        "Greg Zaharchuk",
        "Ghiam Yamin"
    ],
    "DatasetDOI": "TODO: add DOI if available",
    "ReferencesAndLinks": [
        "https://arxiv.org/abs/1903.07988",
        "https://stanfordaimi.azurewebsites.net/datasets/ae0182f1-d5b6-451a-8177-d1f39f01"
    ],
    "HowToAcknowledge": "Please cite Gr\u00f8vik et al., JMRI 2019;51(1):175-182 and the Stanford AIMI dataset page.",
    "GeneratedBy": [
        {
            "Name": "BIDSifier (metadata scaffold only)",
            "Version": "0.1",
            "Description": "Top-level BIDS metadata placeholders; verify and update."
        }
    ],
    "SourceDatasets": [
        {
            "URL": "https://stanfordaimi.azurewebsites.net/datasets/ae0182f1-d5b6-451a-8177-d1f39f01",
            "Version": "unknown",
            "Description": "156 multisequence brain MRI studies; training subset includes segmentations."
        }
    ]
}
"""

    state = {
        "dataset_xml": dataset_xml,
        "publication": publication,
        "readme": readme,
        "output_root": output_root,
        "provider": provider,
        "model": model,
        "step": step,
        "prompt": prompt,
    }
    return response, state


def retry_response(last_state: Optional[Dict[str, Any]]) -> str:
    """
    Retry the LLM call using the last set of inputs.

    Parameters
    ----------
    last_state : dict or None
        A dictionary containing the last used `dataset_xml`, `output_root`,
        `publication`, `readme`, `provider`, `model`, `step`, and `prompt`.
        If `None`, no previous generation is available.

    Returns
    -------
    str
        The simulated LLM response or an informational message if no
        previous state is available.
    """
    if not last_state:
        return "Note: Nothing to retry yet. Please generate a response first."

    dataset_xml = last_state.get("dataset_xml", "")
    publication = last_state.get("publication", "")
    readme = last_state.get("readme", "")
    output_root = last_state.get("output_root", "")
    provider = last_state.get("provider", "")
    model = last_state.get("model", "")
    step = last_state.get("step", "")
    prompt = last_state.get("prompt", "")

    # Reuse the same function for consistency; we ignore the returned state.
    response, _ = generate_response(
        dataset_xml=dataset_xml,
        publication=publication,
        readme=readme,
        output_root=output_root,
        provider=provider,
        model=model,
        step=step,
        prompt=prompt,
    )
    return response


def split_shell_commands(text: str) -> list[str]:
    """
    Split a multi-line shell script into individual commands.

    Each non-empty line is treated as a separate command, except when a line
    ends with a backslash (\\).

    Parameters
    ----------
    text : str
        Multi-line string containing shell commands.

    Returns
    -------
    list of str
        The list of shell commands to execute.
    """
    commands: list[str] = []
    current: str = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            # Skip empty lines
            continue

        if current:
            # We're in the middle of a continued command
            if line.endswith("\\"):
                current += " " + line[:-1].rstrip()
            else:
                current += " " + line
                commands.append(current)
                current = ""
        else:
            # Start of a new command
            if line.endswith("\\"):
                current = line[:-1].rstrip()
            else:
                commands.append(line)

    if current:
        # Trailing continued command without a final line
        commands.append(current)

    return commands


def confirm_output(
    output_root: str,
    llm_output: str,
    step: str,
    progress_value: int,
) -> Tuple[str, str, str, int]:
    """
    Confirm the current LLM output and advance the workflow.

    Parameters
    ----------
    output_root : str
        The output directory path.
    llm_output : str
        The text currently shown in the LLM output area.
    step : str
        Name of the selected BIDS conversion step.
    progress_value : int
        Current numeric progress value (number of confirmed steps).

    Returns
    -------
    status_message : str
        A short confirmation message for display in the UI (includes any
        script/command output).
    new_step : str
        The new selected BIDS step (advances to the next one if available).
    new_llm_output : str
        The updated LLM output (cleared on success).
    new_progress : int
        Updated numeric progress value.
    """
    if not llm_output.strip():
        return (
            "⚠️ There is no LLM output to confirm.",
            step,
            llm_output,
            progress_value,
        )
    if not step:
        return (
            "⚠️ Please select a BIDS step before confirming.",
            step,
            llm_output,
            progress_value,
        )
    if not output_root.strip():
        return (
            "⚠️ Please provide an output root before confirming.",
            step,
            llm_output,
            progress_value,
        )

    # Determine current step index.
    try:
        idx = STEP_KEYS.index(step)
    except ValueError:
        idx = -1

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    config = BIDS_STEPS.get(step, {})
    output_file = config.get("output_file")
    status_detail = ""

    # CONFIRM ACTIONS
    if output_file:
        # If the step has an associated output_file, just write the file.
        path = root / output_file
        path.write_text(llm_output)
        status_detail = f"\n\nWrote file: `{path}`"
    else:
        # No output_file: interpret the LLM output as either a Python script
        # or a general shell command.
        lines = llm_output.splitlines()
        first_line = lines[0].strip() if lines else ""
        second_line = lines[1].strip() if len(lines) > 1 else ""

        if first_line.startswith("#!/usr/bin/env python"):
            # Treat the entire output as a Python script and run it with
            # arguments from the second line.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as tmp:
                tmp.write(llm_output)
                script_path = tmp.name

            cmd = ["python", script_path]
            if second_line:
                # Second line is interpreted as command line arguments.
                cmd.extend(shlex.split(second_line))

            proc = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
            )

            # Optional: clean up the temporary script file.
            try:
                Path(script_path).unlink(missing_ok=True)
            except Exception:
                pass

            cmd_str = " ".join(shlex.quote(c) for c in cmd)
            status_detail = (
                f"\n\nExecuted Python script: `{cmd_str}`\n"
                f"Exit code: {proc.returncode}\n\n"
                f"Stdout:\n{proc.stdout}\n\nStderr:\n{proc.stderr}"
            )
        else:
            # Treat the LLM output as one or more shell commands, one per line,
            # supporting backslash continuations.
            commands = split_shell_commands(llm_output)
            if not commands:
                status_detail = "\n\n(No shell commands to execute.)"
            else:
                details = []
                for cmd in commands:
                    proc = subprocess.run(
                        cmd,
                        shell=True,
                        cwd=str(root),
                        capture_output=True,
                        text=True,
                    )
                    details.append(
                        f"\n\nExecuted shell command:\n`{cmd}`\n"
                        f"Exit code: {proc.returncode}\n\n"
                        f"Stdout:\n{proc.stdout}\n\nStderr:\n{proc.stderr}"
                    )
                status_detail = "".join(details)

    # Update progress: at least current step index + 1.
    if idx >= 0:
        new_progress = max(progress_value, idx + 1)
    else:
        new_progress = progress_value

    # Decide next step and status message.
    if idx >= 0 and idx < NUM_STEPS - 1:
        new_step = STEP_KEYS[idx + 1]
        status = (
            f"✅ LLM output confirmed for step: '{step}'. "
            f"Moving to next step: '{new_step}'."
        )
    elif idx == NUM_STEPS - 1:
        new_step = step
        status = "🎉 All BIDS steps completed!!"
        new_progress = NUM_STEPS
    else:
        # Fallback if step is unknown.
        new_step = step
        status = f"✅ LLM output confirmed for step: '{step}'."

    # Clear the LLM output after successful confirmation.
    new_llm_output = ""

    # Append any details from script/command execution.
    status = status + status_detail

    return status, new_step, new_llm_output, new_progress


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
    status : str
        "pass" if all files are BIDS-compliant, otherwise "fail".
    """
    if not output_root.strip():
        return (
            "⚠️ Please provide an output root before running the BIDS validator.",
            "fail",
        )

    root = Path(output_root)
    if not root.exists():
        return (
            f"⚠️ Output root `{output_root}` does not exist. Nothing to validate.",
            "fail",
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
            "fail",
        )

    summary = (
        f"Validated {valid_count + invalid_count} files: "
        f"{valid_count} OK, {invalid_count} NOT BIDS."
    )
    bullet_lines = "\n".join(f"- `{line}`" for line in lines)

    report = f"### BIDS Validator report\n\n{bullet_lines}\n\n**Summary:** {summary}"

    status_flag = "pass" if invalid_count == 0 and valid_count > 0 else "fail"
    # The value changes each time so .change triggers every time
    status_token = f"{status_flag}:{time.time()}"
    return report, status_token


with gr.Blocks(
    title="OpenBIDSifier (Demo)",
    theme=gr.themes.Citrus(),
    # css=load_css(),
    head="""
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.2/dist/confetti.browser.min.js"></script>
    """,
    # head=f"""
    # <script>
    # {CONFETTI_JS}
    # </script>
    # """,
) as demo:
    gr.Image(
        value="images/bh_logo.png",
        show_label=False,
        height=80,
        elem_id="bh_logo",
    )

    gr.Markdown(
        """
        # OpenBIDSifier (Demo)

        Interface to help convert neuroimaging datasets to BIDS aided by LLMs.
        """
    )

    dataset_xml_input = gr.Textbox(
        label="Dataset XML",
        placeholder="brainmets_structure.xml content",
        lines=4,
    )

    publication_input = gr.Textbox(
        label="Publication",
        placeholder="brainmets-publication-snippets.md content",
        lines=4,
    )

    readme_input = gr.Textbox(
        label="README",
        placeholder="brainmets-readme.md content",
        lines=4,
    )

    with gr.Accordion("LLM settings (advanced)", open=False):
        provider_input = gr.Dropdown(
            label="Provider",
            choices=["openai"],
            value="openai",
            allow_custom_value=False,
        )
        model_input = gr.Dropdown(
            label="Model",
            choices=["gpt-5"],
            value="gpt-5",
        )

    output_root_input = gr.Textbox(
        label="Output root",
        placeholder="brainmets-bids",
        lines=1,
    )

    step_dropdown = gr.Dropdown(
        label="BIDS step",
        choices=STEP_KEYS,
        value=STEP_KEYS[0],
        info="Select the current step in the BIDS conversion workflow.",
        allow_custom_value=False,
    )

    progress_bar = gr.Slider(
        label="Progress through BIDS steps",
        minimum=0,
        maximum=NUM_STEPS,
        step=1,
        value=0,
        interactive=False,
    )

    # Accordion with editable prompt
    with gr.Accordion("Prompt", open=False):
        prompt_editor = gr.Textbox(
            label="Prompt to send to the LLM",
            lines=10,
            placeholder="The default prompt for the selected step will appear here.",
        )

    # Generate / Retry / Confirm buttons
    generate_button = gr.Button("Call LLM", variant="primary")

    # LLM output area
    llm_output = gr.Textbox(
        label="LLM output",
        lines=10,
        interactive=True,
    )

    extra_info = gr.Textbox(
        label='Extra information (Press "Retry" after filling this)',
        lines=1,
        interactive=True,
    )

    with gr.Row():
        with gr.Column():
            retry_button = gr.Button("Retry", variant="secondary")
        with gr.Column():
            confirm_button = gr.Button("Confirm", variant="primary")

    # Extra BIDS Validator button
    bids_validator_button = gr.Button("Run BIDS Validator", variant="secondary")

    # Status / confirmation / validation message
    status_msg = gr.Markdown("")
    validation_status = gr.Textbox(visible=False)

    # Internal state to store last generation inputs for Retry
    last_state = gr.State(value=None)

    def on_step_change(step: str, dataset_xml: str, output_root: str):
        """
        Update the prompt editor and the Confirm button when the user
        selects a step.

        Parameters
        ----------
        step : str
            Selected BIDS step.
        dataset_xml : str
            Current dataset XML content/path.
        output_root : str
            Current output root path.

        Returns
        -------
        prompt : str
            The default prompt for the selected step.
        button_update : dict
            Update object for the confirm button (label changes depending
            on the selected step).
        """
        prompt = get_default_prompt(step, dataset_xml, output_root)

        # Change button label to "Run" only for the Validation step.
        # if step.startswith("3. Validation"):
        #     button_update = gr.update(value="Run")
        # else:
        # No need to change the button
        button_update = gr.update(value="Confirm")

        return prompt, button_update

    step_dropdown.change(
        fn=on_step_change,
        inputs=[step_dropdown, dataset_xml_input, output_root_input],
        outputs=[prompt_editor, confirm_button],
    )

    generate_button.click(
        fn=generate_response,
        inputs=[
            dataset_xml_input,
            publication_input,
            readme_input,
            extra_info,
            output_root_input,
            provider_input,
            model_input,
            step_dropdown,
            prompt_editor,
        ],
        outputs=[llm_output, last_state],
    )

    retry_button.click(
        fn=retry_response,
        inputs=[last_state],
        outputs=[llm_output],
    )

    confirm_button.click(
        fn=confirm_output,
        inputs=[output_root_input, llm_output, step_dropdown, progress_bar],
        outputs=[status_msg, step_dropdown, llm_output, progress_bar],
    )

    bids_validator_button.click(
        fn=run_bids_validation,
        inputs=[output_root_input],
        outputs=[status_msg, validation_status],
    )

    # Trigger confetti in the browser when validation passes
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
