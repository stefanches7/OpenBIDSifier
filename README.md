#  Open Data BIDSifier: LLM-based BIDSifier and metadata harmonization for non-BIDS datasets

This is the repository for an exploratory project Open Data BIDSifier, inspired by the accelerating progress of LLMs and their potential to automate format conversion for multi-study neuroimaging and clinical datasets.

## Motivation

## Project Aim

The goal of this project is to explore if LLM-based workflow machines (sometimes calles "AI agents) can meaningfully assist in metadata harmonization, data format transformation and data preprocessing tasks that precede AI model inference or training. These tasks are time-consuming yet essential to reproducible, large-scale neuroimaging research.

While the BIDS standard ensures interoperability, there are some datasets for which no BIDS annotation is available. This is a "dead data" which can not be used on-par with BIDS datasets.

In this proof-of-concept, we aim to determine whether a coordinated system of AI agents can reliably execute these operations and produce AI-ready dataset collections, similar to those hosted on platforms like HuggingFace Datasets: [OpenMind](https://huggingface.co/datasets/AnonRes/OpenMind), with minimal human intervention.

## The Vision

If successful, Open Data BIDSifier will serve as a foundation for harmonizing multi-study datasets from open data, making sure these are immediately usable for machine learning and statistical analysis.

## Rough Work Plan

We will focus on data and metadata harmonization and evaluation of the results. For evaluation, a manual baseline would be prepared by humans.

### 1. Metadata harmonization

Zenodo and other data portals are rich on non-BIDS neuroimaging data. For a primer, these datasets are suggested: 
- [UniToBrain](https://zenodo.org/records/5109415)
- [Cranial CT of 1 patient](https://zenodo.org/records/16816)
- [BraTS 2020](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
- [Macaque neurodevelopment database](https://data.kitware.com/#collection/54b582c38d777f4362aa9cb3)

The goal is to harmonize them with the following BIDS datasets:
- [Sexing the parental brain in shopping: an fMRI study](doi:10.18112/openneuro.ds006844.v1.0.0)
- [Cross-modal Hierarchical Control](https://openneuro.org/datasets/ds006628/versions/1.0.1)

and really any other OpenNeuro dataset.

6 WPs are the following
Harmonizing **metadata** *with LLM based tools*:
1. Annotation column names (from non-BIDS to BIDS)
2. File structure
3. Study metadata (fetching from repository HTMLs too)

For the LLM-assisted workflow, following **tools** are suggested: Github Copilot in VS Code, LLMAnything. 
Pick **any LLM** really, smaller LLMs tend to hallucinate more, therefore it is more interesting if they can make it too!
Suggestions bigger LLMs: GPT-5, Claude, Kimi-K2, DeepSeek-R1
Suggestions smaller LLMs: SmoLM, LLaMA-7B, Qwen-7B

Harmonizing **metadata** *by hand*:
4. Annotation column names (from non-BIDS to BIDS)
5. File structure
6. Study metadata (fetching from repository HTMLs too)

For the manual harmonization, an IDE with Python / R is useful; as well as OpenRefine, an open-source tool for working with tabular data.


Record the problems and the working time for both manual and LLM assisted harmonization.

Time planned: ~5 hours working time are planned for this step.

### 2. Evaluation of the harmonized metadata

Let's see how well the harmonization went!
Use [BIDS validator](https://bids-standard.github.io/bids-validator/) on the newly-BIDS datasets, and report whether:
1. The manual harmonization is BIDS compliant.
2. The semi-automatic harmonization is BIDS compliant.

3. Assess the differences in the harmonized metadata.
4. Try to "stack" BIDS converted datasets on old BIDS datasets, and report the errors.

Time planned: ~4 hours.

### 3. Try in action

Use [ResEncL](https://huggingface.co/AnonRes/ResEncL-OpenMind-MAE) or another model of your choice with JAX, and feed it the resulting harmonized dataset. Analyse and record the bugs. 

Time planned: ~4 hours.

### Teams and Participants / Skills

Anyone from master students to experienced scientists is welcome to join. The project will involve a lot of tabular data analysis and scripting, for this, coding experience in Python and R and Shell experience are useful.
For working with LLM agents, prompt engineering skills can be useful, though can also be acquired in this project.

### References / Prior reading

- [What is an AI agent](https://blog.langchain.com/how-to-think-about-agent-frameworks/)
- [Prompting guide from Meta](https://www.llama.com/docs/how-to-guides/prompting/)
- [Metadata Harmonization from Biological Datasets with Language Models](https://doi.org/10.1101/2025.01.15.633281)
- [Unifying Heterogeneous Medical Images Using Large Language Models](10.5281/zenodo.15480675)
