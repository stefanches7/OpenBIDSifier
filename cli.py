import argparse
import logging
import os
import re
import sys
from typing import List, Optional
from pathlib import Path
from logging_utils import setup_logging

from agent import BIDSifierAgent


def _read_optional(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_commands_from_markdown(markdown: str) -> List[str]:
    """Extract the first bash/sh fenced code block and return one command per line."""
    pattern = re.compile(r"```(?:bash|sh)\n(.*?)```", re.DOTALL | re.IGNORECASE)
    m = pattern.search(markdown)
    if not m:
        return []
    block = m.group(1)
    commands: List[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        commands.append(line)
    return commands


def _print_commands(commands: List[str]) -> None:
    if not commands:
        print("(No commands detected in fenced bash block.)")
        return
    print("-----"*10)

    print("COMMANDS TO EXECUTE:")
    
    print("-----"*10)
    for c in commands:
        print(f"  {c}")


def prompt_yes_no(question: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    ans = input(f"{question} {suffix} ").strip().lower()
    if not ans:
        return default
    return ans in {"y", "yes"}


def short_divider(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")
    
def enter_feedback_loop(agent: BIDSifierAgent, context: dict, logger: Optional[logging.Logger] = None) -> dict:
    feedback = input("\nAny comments or corrections to the summary? (press Enter to skip): ").strip()
    while feedback:
        if logger:
            logger.info("User feedback: %s", feedback)
        context["user_feedback"] += feedback
        agent_response = agent.run_query(feedback)
        print(agent_response)
        feedback = input("\nAny additional comments or corrections? (press Enter to skip): ").strip()
    return context

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bidsifier",
        description="Interactive LLM assistant to convert a dataset into BIDS via stepwise shell commands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-xml", dest="dataset_xml_path", help="Path to dataset structure XML", required=False)
    parser.add_argument("--readme", dest="readme_path", help="Path to dataset README file", required=False)
    parser.add_argument("--publication", dest="publication_path", help="Path to a publication/notes file", required=False)
    parser.add_argument("--output-root", dest="output_root", help="Target BIDS root directory", required=True)
    parser.add_argument("--provider", dest="provider", help="Provider name or identifier, default OpeanAI", required=False, default="openai")
    parser.add_argument("--model", dest="model", help="Model name to use", default=os.getenv("BIDSIFIER_MODEL", "gpt-4o-mini"))
    parser.add_argument("--project", dest="project", help="Project name for log file prefix", required=False)
    # Execution is intentionally disabled; we only display commands.
    # Keeping --dry-run for backward compatibility (no effect other than display).
    parser.add_argument("--dry-run", dest="dry_run", help="Display-only (default behavior)", action="store_true")

    args = parser.parse_args(argv)

    project_name = args.project or Path(args.output_root).name or Path(os.getcwd()).name
    logger, _listener = setup_logging(project_name=project_name)
    logger.info("Initialized logging for project '%s'", project_name)

    dataset_xml = _read_optional(args.dataset_xml_path)
    readme_text = _read_optional(args.readme_path)
    publication_text = _read_optional(args.publication_path)

    context = {
        "dataset_xml": dataset_xml,
        "readme_text": readme_text,
        "publication_text": publication_text,
        "output_root": args.output_root,
        "user_feedback": "",
    }

    command_env = {
        "OUTPUT_ROOT": args.output_root,
    }
    if args.dataset_xml_path:
        command_env["DATASET_XML_PATH"] = os.path.abspath(args.dataset_xml_path)
    if args.readme_path:
        command_env["README_PATH"] = os.path.abspath(args.readme_path)
    if args.publication_path:
        command_env["PUBLICATION_PATH"] = os.path.abspath(args.publication_path)

    agent = BIDSifierAgent(provider=args.provider, model=args.model)

    short_divider("Step 1: Understand dataset")
    summary = agent.run_step("summary", context)
    print(summary)
    logger.info(summary)
    logger.info("Summary step completed (length=%d chars)", len(summary))
    context = enter_feedback_loop(agent, context, logger)
    if not prompt_yes_no("Proceed to create BIDS root?", default=True):
        logger.info("User aborted after summary step.")
        return 0
    
    short_divider("Step 2: Propose commands to create metadata files")
    meta_plan = agent.run_step("create_metadata", context)
    print(meta_plan)
    cmds = parse_commands_from_markdown(meta_plan)
    _print_commands(cmds)
    logger.info("Metadata plan produced %s", cmds)
    logger.info("Metadata plan produced %d commands", len(cmds))
    context = enter_feedback_loop(agent, context, logger)
    if not prompt_yes_no("Proceed to create empty BIDS structure?", default=True):
        logger.info("User aborted after metadata plan.")
        return 0

    short_divider("Step 3: Propose commands to create dataset structure")
    struct_plan = agent.run_step("create_structure", context)
    print(struct_plan)
    cmds = parse_commands_from_markdown(struct_plan)
    _print_commands(cmds)
    logger.info("Structure plan produced %s", cmds)
    logger.info("Structure plan produced %d commands", len(cmds))
    context = enter_feedback_loop(agent, context, logger)
    if not prompt_yes_no("Proceed to propose renaming/moving?", default=True):
        logger.info("User aborted after structure plan.")
        return 0

    short_divider("Step 4: Propose commands to rename/move files")
    move_plan = agent.run_step("rename_move", context)
    print(move_plan)
    cmds = parse_commands_from_markdown(move_plan)
    _print_commands(cmds)
    logger.info("Rename/move plan produced %s", cmds)
    logger.info("Rename/move plan produced %d commands", len(cmds))
    context = enter_feedback_loop(agent, context, logger)

    print("\nAll steps completed. Commands were only displayed - use them manually")
    logger.info("All steps completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
