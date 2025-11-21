from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
import dspy

import prompts


class BIDSifierAgent:
	"""Wrapper around OpenAI chat API for step-wise BIDSification."""

	def __init__(self, *, provider: Optional[str] = None, model: Optional[str] = None, openai_api_key: Optional[str] = None, temperature: float = 0.2):
		load_dotenv()
		
		if provider=="openai":
			if model == "gpt-5": #reasoning model that requires special handling
				temperature = 1.0
				lm = dspy.LM(f"{provider}/{model}", api_key=openai_api_key or os.getenv("OPENAI_API_KEY"), temperature = temperature, max_tokens = 40000)
			else:
				lm = dspy.LM(f"{provider}/{model}", api_key=openai_api_key or os.getenv("OPENAI_API_KEY"), temperature = temperature, max_tokens = 10000)
		else:
			lm = dspy.LM(f"{provider}/{model}", api_key="", max_tokens=10000)

      

		dspy.configure(lm=lm)
		self.llm = lm
		self.model = model or os.getenv("BIDSIFIER_MODEL", "gpt-4o-mini")
		self.temperature = temperature
 
	def _build_user_prompt(self, step: str, context: Dict[str, Any]) -> str:
		dataset_xml = context.get("dataset_xml")
		readme_text = context.get("readme_text")
		publication_text = context.get("publication_text")
		output_root = context.get("output_root", "./bids_output")

		if step == "summary":
			return prompts.summarize_dataset_prompt(
				dataset_xml=dataset_xml,
				readme_text=readme_text,
				publication_text=publication_text,
			)
		if step == "create_metadata":
			return prompts.create_metadata_prompt(
				output_root=output_root,
				dataset_xml=dataset_xml,
				readme_text=readme_text,
				publication_text=publication_text,
			)
		if step == "create_structure":
			return prompts.create_structure_prompt(
				output_root=output_root,
				dataset_xml=dataset_xml,
				readme_text=readme_text,
				publication_text=publication_text,
			)
		if step == "rename_move":
			return prompts.rename_and_move_prompt(
				output_root=output_root,
				dataset_xml=dataset_xml,
				readme_text=readme_text,
				publication_text=publication_text,
			)
		raise ValueError(f"Unknown step: {step}")

	def run_step(self, step: str, context: Dict[str, Any]) -> str:
		system_msg = prompts.system_prompt()
		user_msg = self._build_user_prompt(step, context)
		resp = self.llm(
			messages=[
				{"role": "system", "content": system_msg},
				{"role": "user", "content": user_msg},
			],
			temperature=self.temperature,
		)
		return resp[0]

	def run_query(self, query: str) -> str:
		system_msg = prompts.system_prompt()
		resp = self.llm(
			messages=[
				{"role": "system", "content": system_msg},
				{"role": "user", "content": query},
			],
			temperature=self.temperature,
		)
		return resp[0]


__all__ = ["BIDSifierAgent"]


