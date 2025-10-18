# Contributing to AI-Assisted Neuroimaging Data Harmonization

Thank you for considering contributing to this project! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, etc.)
- Any relevant code snippets or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:

- A clear, descriptive title
- Detailed description of the proposed feature
- Use cases and benefits
- Any relevant examples or mockups

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Add or update tests as needed
5. Update documentation
6. Commit your changes (`git commit -m 'Add some feature'`)
7. Push to the branch (`git push origin feature/your-feature-name`)
8. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/AI-assisted-Neuroimaging-harmonization.git
cd AI-assisted-Neuroimaging-harmonization

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install development tools
pip install pytest pytest-cov black flake8 mypy
```

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and concise
- Add type hints where appropriate

### Format your code

```bash
# Auto-format with black
black src/

# Check with flake8
flake8 src/

# Type checking with mypy
mypy src/
```

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=ai_neuro_wrangler tests/

# Run specific test
pytest tests/test_specific.py
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions and classes
- Update configuration examples if needed
- Add usage examples for new features

## Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (Add, Fix, Update, etc.)
- Keep the first line under 50 characters
- Add detailed description if needed

Examples:
```
Add support for DICOM format

Fix memory leak in volume normalizer

Update documentation for quality control module
```

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Assume good intentions

## Questions?

If you have questions, feel free to:

- Open an issue with the "question" label
- Reach out to the maintainers
- Check existing documentation and issues

Thank you for contributing! 🎉
