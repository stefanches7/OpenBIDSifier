"""
Prompt templates for the BIDSifier assistant.

Contract expected by the CLI:
- Exactly one bash code block (```bash ... ```), one command per line, no inline comments.
- Prefer safe operations: mkdir -p, cp -n; avoid destructive actions unless explicitly stated.
- Use env vars when present: $OUTPUT_ROOT, $DATASET_XML_PATH, $README_PATH, $PUBLICATION_PATH.
"""

from typing import Optional


SYSTEM_PROMPT = (
	"You are BIDSifier, an LLM assistant that proposes careful, incremental shell commands "
	"to convert non-standard neuroimaging datasets into BIDS. Be conservative and explicit."
)


def _ctx(dataset_xml: Optional[str], readme_text: Optional[str], publication_text: Optional[str]) -> str:
	parts = []
	if dataset_xml:
		parts.append("[Dataset XML]\n" + dataset_xml.strip())
	if readme_text:
		parts.append("[README]\n" + readme_text.strip())
	if publication_text:
		parts.append("[Publication]\n" + publication_text.strip())
	return "\n\n".join(parts) if parts else "[No additional context provided]"


def system_prompt() -> str:
	return SYSTEM_PROMPT


def summarize_dataset_prompt(*, dataset_xml: Optional[str], readme_text: Optional[str], publication_text: Optional[str]) -> str:
	return f"""
Step 1/4 — Understand the dataset and produce a short summary.

Requirements:
- 8–15 concise bullets covering subjects/sessions, modalities (T1w/T2w/DWI/fMRI/etc.), tasks, naming patterns, id conventions.
- Call out uncertainties or missing info explicitly.
- Do not propose any commands in this step.

Context:\n{_ctx(dataset_xml, readme_text, publication_text)}

Output:
- One short paragraph (<=4 sentences) then bullets. End with open questions for the user if any.
"""




def create_metadata_prompt(*, output_root: str, dataset_xml: Optional[str], readme_text: Optional[str], publication_text: Optional[str]) -> str:
	return f"""
Step 2/4 — Propose commands to create required BIDS metadata files.

Must include:
- dataset_description.json (Name, BIDSVersion, License if known)
- participants.tsv and participants.json (headers and column descriptions; can be placeholders)
- README and LICENSE (best guess or TODO)
- Task/event placeholders if task fMRI is suspected

Constraints:
- Use $OUTPUT_ROOT if present, else {output_root}
- Create without overwriting existing content; use here-docs or echo safely. If unsure, add TODO markers.

Context:\n{_ctx(dataset_xml, readme_text, publication_text)}

Output:
- Short rationale bullets, then a single fenced bash block with commands only.
"""


def create_structure_prompt(*, output_root: str, dataset_xml: Optional[str], readme_text: Optional[str], publication_text: Optional[str]) -> str:
	return f"""
Step 3/4 — Propose commands to create the BIDS directory structure.

Goals:
- Infer subjects, sessions, and modalities; create sub-<label>/, optional ses-<label>/, and modality folders (anat, dwi, func, fmap, etc.).
- Do not move/copy raw files yet; create empty structure only.

Constraints:
- Use $OUTPUT_ROOT if present, else {output_root}
- Use mkdir -p.

Context:\n{_ctx(dataset_xml, readme_text, publication_text)}

Output:
- One plan then a single fenced bash block with commands.
"""


def rename_and_move_prompt(*, output_root: str, dataset_xml: Optional[str], readme_text: Optional[str], publication_text: Optional[str]) -> str:
	return f"""
Step 4/4 — Propose commands to rename and move files into the BIDS structure.

Requirements:
- Map original names to BIDS filenames; demonstrate patterns (e.g., with find/xargs) carefully.
- Prefer non-destructive copy (cp -n). Use mv only if explicitly stated by the user.
- Include TODOs for ambiguous mappings; split into small chunks to facilitate review.

Constraints:
- Target $OUTPUT_ROOT (or {output_root}).
- Reference inputs via env vars when possible.

Context:\n{_ctx(dataset_xml, readme_text, publication_text)}

Output:
- A brief mapping summary (text) followed by a single fenced bash block with commands only.
"""

