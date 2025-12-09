# Contributing to CapabilityMesh

Thank you for your interest in contributing to CapabilityMesh! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [existing issues](https://github.com/scionoftech/capabilitymesh/issues) to avoid duplicates.

When creating a bug report, include:
- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Python version and CapabilityMesh version
- Code samples or error messages
- Any relevant logs or screenshots

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:
- A clear and descriptive title
- Detailed description of the proposed feature
- Use cases and examples
- Why this enhancement would be useful
- Potential implementation approach (optional)

### Contributing Code

We welcome pull requests! Here's how to contribute code:

1. **Fork the Repository**
   ```bash
   git clone https://github.com/scionoftech/capabilitymesh.git
   cd capabilitymesh
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make Your Changes**
   - Write code following our coding standards
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   pytest tests/ -v
   ruff check capabilitymesh/
   black --check capabilitymesh/ tests/
   mypy capabilitymesh/ --ignore-missing-imports
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your feature branch
   - Fill in the PR template

## Development Setup

### Prerequisites

- Python 3.9 or higher
- git
- pip

### Installation

1. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/capabilitymesh.git
   cd capabilitymesh
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

4. Verify installation:
   ```bash
   pytest tests/
   ```

### Optional Framework Dependencies

To work on specific framework integrations:

```bash
# For CrewAI integration
pip install -e .[crewai]

# For AutoGen integration
pip install -e .[autogen]

# For LangGraph integration
pip install -e .[langgraph]

# For all frameworks
pip install -e .[frameworks]
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting (line length: 100)
- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Use type hints for all function signatures

### Code Formatting

Before committing, format your code:

```bash
# Format code
black capabilitymesh/ tests/ examples/

# Check formatting
black --check capabilitymesh/ tests/

# Lint code
ruff check capabilitymesh/

# Type check
mypy capabilitymesh/ --ignore-missing-imports
```

### Documentation

- Add docstrings to all public modules, classes, and functions
- Use Google-style docstrings
- Update README.md if adding new features
- Add examples for new functionality

Example docstring:

```python
def extract_capabilities(self, agent: Any) -> List[Capability]:
    """Extract capabilities from a framework agent.

    Args:
        agent: The framework-specific agent instance

    Returns:
        List of extracted ACDP capabilities

    Raises:
        ValueError: If agent is invalid
    """
    pass
```

### Naming Conventions

- Classes: `PascalCase` (e.g., `AgentIdentity`)
- Functions/methods: `snake_case` (e.g., `extract_capabilities`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`)
- Private members: prefix with `_` (e.g., `_internal_method`)

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual modules
│   ├── test_capability.py
│   └── test_identity.py
└── integration/    # Integration tests for adapters
    ├── test_crewai_integration.py
    └── test_a2a_integration.py
```

### Writing Tests

- Write tests for all new functionality
- Aim for >80% code coverage
- Use descriptive test names
- Include both positive and negative test cases
- Use pytest fixtures for common setup

Example test:

```python
import pytest
from capabilitymesh import Capability, CapabilityVersion

class TestCapability:
    """Tests for Capability class."""

    @pytest.fixture
    def sample_capability(self):
        """Create a sample capability for testing."""
        return Capability(
            id="cap-test",
            name="test-capability",
            version=CapabilityVersion(major=1, minor=0, patch=0),
            # ... other fields
        )

    def test_create_capability(self, sample_capability):
        """Test creating a capability."""
        assert sample_capability.id == "cap-test"
        assert sample_capability.name == "test-capability"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_capability.py -v

# Run with coverage
pytest tests/ -v --cov=capabilitymesh --cov-report=html

# Run integration tests only
pytest tests/integration/ -v
```

## Pull Request Process

### PR Checklist

Before submitting a pull request, ensure:

- [ ] Code follows the style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated (README, docstrings, examples)
- [ ] Changelog updated (CHANGELOG.md)
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages follow conventional commits format

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(crewai): add support for custom tool extraction
fix(a2a): resolve DID uniqueness issue in adapter
docs(readme): update installation instructions
test(integration): add tests for AutoGen adapter
```

### PR Review Process

1. Automated CI checks will run on your PR
2. Maintainers will review your code
3. Address any requested changes
4. Once approved, your PR will be merged

## Release Process

(For maintainers)

1. Update version in `pyproject.toml` and `capabilitymesh/__init__.py`
2. Update `CHANGELOG.md` with release notes
3. Create a git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions will automatically publish to PyPI

## Areas Looking for Contributors

We especially welcome contributions in these areas:

### High Priority
- [ ] P2P Discovery Engine (mDNS, Gossip, DHT)
- [ ] Negotiation Protocol implementation
- [ ] Trust and Reputation System
- [ ] CLI tools

### Medium Priority
- [ ] Additional transport protocols (gRPC, MQTT)
- [ ] Semantic matching with embeddings
- [ ] Storage backends (Redis, SQLite)
- [ ] More framework integrations (LlamaIndex, Haystack)

### Low Priority
- [ ] Performance optimizations
- [ ] Additional examples and tutorials
- [ ] Documentation improvements
- [ ] UI for capability browsing

## Getting Help

- **Documentation**: [https://scionoftech.github.io/capabilitymesh](https://scionoftech.github.io/capabilitymesh)
- **Issues**: [GitHub Issues](https://github.com/scionoftech/capabilitymesh/issues)
- **Discussions**: [GitHub Discussions](https://github.com/scionoftech/capabilitymesh/discussions)

## Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for contributing to CapabilityMesh! 🎉
