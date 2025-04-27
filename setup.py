import re

from setuptools import Extension, find_packages, setup

# Read the contents of README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Find the `Description` section in the README file
description_pattern = re.compile(
    r"^## Description\s*(.*?)\s*##", re.DOTALL | re.MULTILINE
)
match = description_pattern.search(long_description)
if match:
    # Extract the description text
    description = match.group(1).strip()

# Read the requirements file for the setup dependencies
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Define the C extension module
LempelZivModule_extension = Extension(
    "ComputablePhaseMetrics.ComputableInformationDensity.lempel_ziv_complexity.LempelZivModule",
    sources=[
        "src/ComputablePhaseMetrics/ComputableInformationDensity/lempel_ziv_complexity/LempelZivModule.c"
    ],
)

setup(
    name="ComputablePhaseMetrics",
    version="0.1.0",
    description="Computable information densities as universal indicators for phase transitions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Benjamin H. Andersen, Morten Holm, Simon G. Andersen, Amin Doostmohammadi",
    author_email="-, MortenHolmRepo@pm.me, -, -",
    url="https://github.com/MortenHolmRep/ComputablePhaseMetrics",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.10"
        "Programming Language :: Python :: 3.11"
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
            "mypy",
            "ruff",
            "sphinx",
            "matplotlib",
            "plotly",
        ],
    },
    ext_modules=[LempelZivModule_extension],
    project_urls={
        "Bug Reports": "https://github.com/mortenholmrepo/ComputablePhaseMetrics/issues",
        "Source": "https://github.com/mortenholmrepo/ComputablePhaseMetrics",
        "Documentation": "https://computablephasemetrics.readthedocs.io/",
    },
)
