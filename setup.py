from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-neuro-wrangler",
    version="0.1.0",
    author="AI-assisted Neuroimaging Team",
    description="AI-assisted data wrangling for neuroimaging datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stefanches7/AI-assisted-Neuroimaging-harmonization",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ai-neuro-wrangler=ai_neuro_wrangler.cli:main",
        ],
    },
)
