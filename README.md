#  Open Data BIDSifier: LLM-based BIDSifier and metadata harmonization for non-BIDS datasets

This is the repository for an exploratory project Open Data BIDSifier, inspired by the accelerating progress of LLMs and their potential to automate format conversion for multi-study neuroimaging and clinical datasets.

## Motivation

## Project Aim

The goal of this project is to explore if LLM-based workflow machines (sometimes calles "AI agents) can meaningfully assist in metadata harmonization, data format transformation and data preprocessing tasks that precede AI model inference or training. These tasks are time-consuming yet essential to reproducible, large-scale neuroimaging research.

While the BIDS standard ensures interoperability, there are some datasets for which no BIDS annotation is available. This is a "dead data" which can not be used on-par with BIDS datasets.

In this proof-of-concept, we aim to determine whether a coordinated system of AI agents can reliably execute these operations and produce AI-ready dataset collections, similar to those hosted on platforms like HuggingFace Datasets: [OpenMind](https://huggingface.co/datasets/AnonRes/OpenMind), with minimal human intervention.

## The Vision

If successful, Open Data BIDSifier will serve as a foundation for an AI agent that can identify and harmonize different datasets from open data, making sure these are immediately usable for machine learning and statistical analysis.

## Tools and Structure 

We will use this repository (https://github.com/stefanches7/AI-assisted-Neuroimaging-harmonization) as an intermittant commit place. Please, make yourself familiar with Git and Github. [This intro](https://docs.github.com/de/get-started/start-your-journey/hello-world) can be useful for that.

### Git guidelines

Open a new branch and create Pull requests to the main for the additions. 

### Working with data 

We will work with raw data (Neuroimaging) and annotation / metadata (tabular data).
For Neuroimaging, `nibabel` (.nii file format) and `pydicom` (.dcm file format) are the most advanced Python libraries. 
For working with tabular data and manual harmonization, Python package `pandas` is the standard way; as well as OpenRefine, an open-source tool for working with tabular data.

### LLM usage

For the LLM-assisted workflow, following **tools** are suggested: Github Copilot in VS Code, LLMAnything. 
Pick **any LLM** really, smaller LLMs tend to hallucinate more, therefore it is more interesting if they can make it too!
Suggestions bigger LLMs: GPT-5, Claude, Kimi-K2, DeepSeek-R1
Suggestions smaller LLMs: SmoLM, LLaMA-7B, Qwen-7B

### Coding & Vibe Coding

We will use **Python** to code. I recommend using **Anaconda** package manager as a tool to manage the Python package environments. If you are not sure what the previous 2 sentences really mean, I recommend [reading this intro to Python & Conda](https://www.anaconda.com/topics/choosing-between-anaconda-vs-python#:~:text=Anaconda%20is%20a%20distribution%20that,machine%20learning%2C%20and%20scientific%20computing.)

LLMs can assist in writing code, but can also prove counterproductive and write bad (spaghetti), duplicated and erroneous code. It is instructful to be able check their output and correct it manually.

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
1. Annotation column names (from non-BIDS to BIDS) - working with tabular data
2. File structure
3. Study metadata (fetching from repository HTMLs too)

Harmonizing **metadata** *by hand*:
4. Annotation column names (from non-BIDS to BIDS) - working with tabular data
5. File structure
6. Study metadata (fetching from repository HTMLs too)

Record the problems and the working time for both manual and LLM assisted harmonization.

Time planned: ~4 hours working time are planned for this step.

### 2. Evaluation of the harmonized metadata

Let's see how well the harmonization went!
Use [BIDS validator](https://bids-standard.github.io/bids-validator/) on the newly-BIDS datasets, and report whether:
1. The manual harmonization is BIDS compliant.
2. The semi-automatic harmonization is BIDS compliant.

3. Assess the differences in the harmonized metadata.
4. Try to "stack" BIDS converted datasets on old BIDS datasets, and report the errors.

Time planned: ~3 hours.

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
