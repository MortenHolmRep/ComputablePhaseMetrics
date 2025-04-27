# ComputablePhaseMetrics

## Description
This module provides functionality to compute Computable Information Densities (CID), 
a proposed universal indicator metric for identifying phase transitions in complex systems out of equilibrium. 
CID is a measure derived from algorithmic information theory and is used to analyze 
the structural and dynamical properties of systems undergoing phase transitions. 

The module includes methods for calculating CID values and interpreting them in the 
context of phase transition phenomena, making it a valuable tool for researchers 
studying criticality and emergent behavior in various scientific domains.

## Installation Guide

To install the package in editable mode with development dependencies, use the following command:

```bash
pip install -e .[dev]
```

Make sure you have `pip` installed and are in the root directory of the project before running the command.

### Building the C Extension
The package includes a C extension module (`LempelZivModule`) for improved performance. The installation process will automatically build this module.

## Contributing

We welcome contributions to improve this project! Please follow these guidelines to ensure a smooth contribution process.

### GitHub Issues

Use GitHub issues for tracking and discussing requests and bugs. If there is anything you'd wish to contribute:
- Create a new issue describing what you would like to work on, or
- Assign an existing open issue to yourself to take ownership of a particular task

Using issues actively ensures transparency and agreement on priorities. This helps avoid situations with development effort going into features outside the project scope or solutions that could be better implemented differently.

### How to Contribute

1. **Fork the repository**
   - Click the "Fork" button at the top right of this repository
   - This creates a copy of the repository in your GitHub account

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR-USERNAME/REPOSITORY-NAME.git
   cd REPOSITORY-NAME
   ```

3. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
   - Develop code in dedicated feature branches on your forked repository
   - Use descriptive branch names (e.g., `add-2d-ising-model`)

4. **Make your changes**
   - Implement your feature or bug fix
   - Add or update tests as necessary
   - Update documentation as needed

5. **Commit your changes**
   ```bash
   git commit -m "Add a descriptive commit message"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Go to the original repository
   - Click "Pull Requests" and then "New Pull Request"
   - Click "compare across forks" and select your fork and branch
   - Add a title and description that clearly explains your changes
   - Submit the pull request

### Pull Request Requirements

To be accepted, pull requests must:
- Pass all automated checks (pending implementation)
- Be reviewed by at least one other contributor

Reviews should check for:
1. Standard Python coding conventions (PEP8)
2. Google-style docstrings and type hinting as necessary
3. Unit tests as necessary
4. Clean coding practices

### Conventions

This repository aims to support Python versions that are actively supported (currently `>=3.10`). 
Standard Python coding conventions should be followed:
- Adhere to PEP8
- Use clean code practices when relevant

### Code Quality

To ensure consistency in code style and adherence to best practices, we **require** that all developers use:
- `docformatter` for docstring formatting
- `mypy` for static type checking
- `ruff` for linting

This can conveniently be done using pre-commit hooks. To set this up:

1. Make sure you have installed the `pre-commit` Python package:
   ```bash
   pip install -e .[dev]
   ```

2. Install the pre-commit hooks:
   ```bash
   pre-commit install
   ```

Then, every time you commit a change, your code and docstrings will automatically be formatted and checked for errors and adherence to PEP8, PEP257, and static typing.
