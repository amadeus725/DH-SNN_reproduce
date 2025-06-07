# Contributing to DH-SNN Reproduction Project

Thank you for your interest in contributing to the DH-SNN reproduction project! This document provides guidelines for contributing to ensure a collaborative and productive environment.

## 🎯 Ways to Contribute

### 🐛 Bug Reports
- Report bugs using GitHub Issues
- Include detailed reproduction steps
- Provide system information (OS, Python version, etc.)
- Include error messages and stack traces

### ✨ Feature Requests
- Suggest new features or improvements
- Explain the use case and benefits
- Discuss implementation ideas

### 🔧 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test coverage improvements

### 📚 Documentation
- API documentation
- Tutorial improvements
- Example code
- Architecture explanations

## 🚀 Getting Started

### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/your-username/DH-SNN_reproduce.git
cd DH-SNN_reproduce

# Add upstream remote
git remote add upstream https://github.com/original-owner/DH-SNN_reproduce.git
```

### 2. Set Up Development Environment
```bash
# Create conda environment
conda create -n dh-snn-dev python=3.9
conda activate dh-snn-dev

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch
```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number
```

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use type hints where applicable
- Write docstrings for all public functions and classes
- Keep functions small and focused

### Code Formatting
We use the following tools for code formatting:

```bash
# Format code
black src/ experiments/ tests/

# Sort imports
isort src/ experiments/ tests/

# Lint code
flake8 src/ experiments/ tests/

# Type checking
mypy src/
```

### Testing
- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### Documentation
- Update docstrings for new/modified functions
- Update README if adding new features
- Add examples for new functionality

## 🔬 Scientific Contributions

### Experiments
- Follow reproducible research practices
- Document experimental setup clearly
- Include statistical significance testing
- Provide configuration files for experiments

### Models and Algorithms
- Include mathematical descriptions
- Reference original papers
- Validate against established benchmarks
- Provide ablation studies

## 📋 Pull Request Process

### 1. Before Submitting
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### 2. PR Description
Include in your PR description:
- **Summary**: Brief description of changes
- **Motivation**: Why this change is needed
- **Changes**: Detailed list of modifications
- **Testing**: How you tested the changes
- **Issues**: Link related issues

### 3. PR Template
```markdown
## Summary
Brief description of the changes.

## Motivation and Context
Why is this change required? What problem does it solve?

## Changes Made
- [ ] Feature/Fix 1
- [ ] Feature/Fix 2
- [ ] Documentation updates

## Testing
- [ ] Unit tests added/updated
- [ ] All tests pass
- [ ] Manual testing performed

## Screenshots (if applicable)
Add screenshots for UI changes.

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No breaking changes (or noted)
```

### 4. Review Process
- Maintainers will review your PR
- Address feedback and make requested changes
- Keep discussions constructive and professional
- Be patient - reviews take time

## 🏷️ Commit Message Guidelines

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples
```
feat(models): add multi-branch dendritic layer

- Implement DendriticDenseLayer with configurable branches
- Add temporal constant initialization
- Include connection masking functionality

Closes #123
```

```
fix(training): resolve gradient explosion in BPTT

- Implement gradient clipping in DH-SRNN
- Add stability checks for temporal dynamics
- Update default learning rate

Fixes #456
```

## 🚫 Code of Conduct

### Our Standards
- Be respectful and inclusive
- Welcome newcomers and different perspectives
- Focus on constructive criticism
- Help others learn and grow

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Inappropriate sexual content

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code reviews and contributions

### Documentation
- Read the [README](README.md) for project overview
- Check [docs/](docs/) for detailed documentation
- Look at existing code for patterns and examples

## 🎖️ Recognition

Contributors will be recognized in:
- README.md acknowledgments
- CONTRIBUTORS.md file
- Release notes for significant contributions

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to DH-SNN! Your efforts help advance spiking neural network research. 🧠✨
