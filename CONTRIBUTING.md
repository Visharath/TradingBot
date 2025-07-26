# Contributing to Unified CNN-LSTM Trading Bot

Thank you for your interest in contributing to the Unified CNN-LSTM Trading Bot! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of machine learning and trading concepts
- Familiarity with TensorFlow/Keras, pandas, and numpy

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork the repo on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/TradingBot.git
   cd TradingBot
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Verify installation**
   ```bash
   pytest tests/ -v
   ```

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Bug fixes, new features, optimizations
- **Documentation**: Improve docs, add examples, write tutorials
- **Testing**: Add test cases, improve coverage
- **Performance**: Optimize existing code

### Finding Issues to Work On

- Look for issues labeled `good first issue` for beginner-friendly tasks
- Issues labeled `help wanted` indicate where we need community support
- Check the project roadmap for larger features we're planning

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line Length**: 88 characters (Black formatter default)
- **String Quotes**: Double quotes preferred
- **Import Organization**: Use isort for consistent import ordering

### Code Formatting

We use automated formatting tools:

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with flake8
flake8 src/ tests/
```

### Type Hints

- All public functions and methods must include type hints
- Use `typing` module for complex types
- Example:
  ```python
  from typing import List, Dict, Optional, Tuple
  
  def process_data(
      data: pd.DataFrame,
      columns: List[str],
      config: Optional[Dict[str, Any]] = None
  ) -> Tuple[np.ndarray, np.ndarray]:
      """Process market data and return features and targets."""
      # Implementation here
  ```

### Docstring Standards

Use Google-style docstrings:

```python
def calculate_indicators(
    data: pd.DataFrame,
    periods: List[int]
) -> Dict[str, np.ndarray]:
    """
    Calculate technical indicators for given periods.
    
    Args:
        data: OHLCV data with required columns
        periods: List of periods for moving averages
        
    Returns:
        Dictionary mapping indicator names to arrays
        
    Raises:
        ValueError: If data is missing required columns
        
    Example:
        >>> data = pd.DataFrame(...)
        >>> indicators = calculate_indicators(data, [10, 20, 50])
        >>> print(indicators.keys())
        dict_keys(['sma_10', 'sma_20', 'sma_50'])
    """
```

## Testing Guidelines

### Test Structure

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark critical functions

### Writing Tests

```python
import pytest
from src.data_processor import DataProcessor

class TestDataProcessor:
    """Test cases for DataProcessor."""
    
    def test_process_data_basic(self, sample_data):
        """Test basic data processing functionality."""
        processor = DataProcessor()
        features, targets, names = processor.process_data(sample_data)
        
        assert isinstance(features, np.ndarray)
        assert len(targets) == len(features)
        assert len(names) == features.shape[1]
    
    @pytest.mark.slow
    def test_process_large_dataset(self, large_dataset):
        """Test processing with large dataset."""
        # Test implementation
```

### Test Markers

Use pytest markers to categorize tests:

- `@pytest.mark.unit`: Unit tests (fast)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Slow tests (>1 second)
- `@pytest.mark.requires_data`: Tests needing external data
- `@pytest.mark.requires_broker`: Tests needing broker connection

### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_processor.py -v
```

## Documentation Guidelines

### Code Documentation

- Every public class, method, and function needs docstrings
- Include examples in docstrings when helpful
- Document complex algorithms and business logic
- Keep documentation up-to-date with code changes

### User Documentation

- Write clear, beginner-friendly tutorials
- Include complete code examples
- Provide troubleshooting sections
- Add diagrams for complex concepts

### Documentation Structure

```
docs/
├── installation.md       # Installation guide
├── architecture.md       # Technical architecture
├── features.md          # Feature documentation
├── api_reference.md     # API documentation
├── tutorials/           # Step-by-step guides
├── examples/            # Code examples
└── troubleshooting.md   # Common issues
```

## Pull Request Process

### Before Submitting

1. **Create an Issue**: For new features, create an issue to discuss the approach
2. **Fork and Branch**: Work on a feature branch, not main
3. **Follow Conventions**: Use clear commit messages and branch names
4. **Test Thoroughly**: Ensure all tests pass and add new tests
5. **Update Documentation**: Update relevant documentation

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass (including new tests for new features)
- [ ] Documentation is updated
- [ ] Type hints are included
- [ ] Docstrings are complete and accurate
- [ ] No breaking changes (or clearly documented)
- [ ] Performance impact is considered

### Commit Message Format

Use conventional commit format:

```
type(scope): short description

Longer description if needed

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `perf`: Performance improvements

Examples:
```
feat(data): add support for cryptocurrency data sources

fix(trading): handle connection timeouts gracefully

docs(readme): update installation instructions for macOS
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: Maintainers review code quality and design
3. **Testing**: Ensure functionality works as expected
4. **Documentation Review**: Check docs are accurate and complete
5. **Final Approval**: Maintainer approves and merges

## Issue Guidelines

### Bug Reports

Include the following information:

```markdown
**Bug Description**
Clear description of the bug.

**Steps to Reproduce**
1. Step one
2. Step two
3. Expected vs actual behavior

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Package version: [e.g., 1.0.0]
- Dependencies: [relevant package versions]

**Additional Context**
- Error messages or logs
- Screenshots if relevant
- Minimal code example
```

### Feature Requests

Include the following information:

```markdown
**Feature Description**
Clear description of the proposed feature.

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed Implementation**
How should this feature work? Include examples if possible.

**Alternatives Considered**
What other approaches did you consider?

**Additional Context**
Any other relevant information.
```

## Development Workflow

### Setting Up Your Development Environment

1. **Clone and Setup**
   ```bash
   git clone https://github.com/YOUR_USERNAME/TradingBot.git
   cd TradingBot
   python -m venv venv
   source venv/bin/activate
   pip install -e .[dev]
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   pytest tests/ -v
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Use the GitHub web interface
   - Fill out the PR template
   - Wait for review and address feedback

### Working with External Dependencies

When adding new dependencies:

1. **Evaluate Necessity**: Is this dependency really needed?
2. **Check License**: Ensure license compatibility
3. **Consider Alternatives**: Look for lighter alternatives
4. **Add to Requirements**: Update requirements.txt
5. **Document**: Update installation docs if needed

### Performance Considerations

- **Profile Before Optimizing**: Use profiling tools to identify bottlenecks
- **Benchmark Changes**: Compare performance before and after
- **Memory Usage**: Monitor memory consumption, especially for large datasets
- **Scalability**: Consider how changes affect performance with large inputs

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Requests**: Code contributions, reviews

### Getting Help

- Check existing documentation and examples
- Search GitHub issues for similar problems
- Create a new issue with detailed information
- Join community discussions for broader questions

### Recognition

Contributors are recognized in:

- README.md contributors section
- Release notes for significant contributions
- GitHub contributor graphs
- Special mentions for outstanding contributions

## Release Process

### Versioning

We use Semantic Versioning (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. Update version number
2. Update CHANGELOG.md
3. Create release notes
4. Tag release in Git
5. Build and publish packages
6. Update documentation

Thank you for contributing to the Unified CNN-LSTM Trading Bot! Your contributions help make algorithmic trading more accessible to everyone.