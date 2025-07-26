# Contributing to Unified CNN-LSTM Trading Bot

We welcome contributions to the Unified CNN-LSTM Trading Bot! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/TradingBot.git
   cd TradingBot
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/Visharath/TradingBot.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, or virtualenv)

### Setup Instructions

1. **Create and activate virtual environment**:
   ```bash
   python -m venv trading-bot-dev
   source trading-bot-dev/bin/activate  # On Windows: trading-bot-dev\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Verify installation**:
   ```bash
   pytest tests/ -v
   python -m src.unified_trading_bot --help
   ```

## Contributing Process

### Before You Start

1. **Check existing issues** to see if your feature/bug is already being worked on
2. **Create an issue** to discuss major changes before implementing
3. **Keep changes focused** - one feature/fix per pull request

### Making Changes

1. **Create a new branch** from `develop`:
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run tests locally**:
   ```bash
   pytest tests/
   black src/ tests/ examples/
   flake8 src/ tests/ examples/
   mypy src/
   ```

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Import sorting**: isort with Black profile
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public classes and functions

### Code Formatting

We use automated tools for consistent formatting:

```bash
# Format code
black src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/

# Lint code
flake8 src/ tests/ examples/

# Type checking
mypy src/
```

### Example Code Style

```python
"""
Module docstring describing the purpose and functionality.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

from src.config import Config


class ExampleClass:
    """
    Example class demonstrating coding standards.
    
    Args:
        config: Configuration object
        symbols: List of trading symbols
    """
    
    def __init__(self, config: Config, symbols: List[str]):
        self.config = config
        self.symbols = symbols
        self._private_attribute = None
    
    def public_method(
        self, 
        data: pd.DataFrame, 
        threshold: float = 0.02
    ) -> Optional[Dict[str, Any]]:
        """
        Example public method with proper typing and documentation.
        
        Args:
            data: Input market data
            threshold: Signal threshold percentage
            
        Returns:
            Dictionary with results or None if failed
            
        Raises:
            ValueError: If data is invalid
        """
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        # Implementation here
        return {"result": "success"}
    
    def _private_method(self) -> None:
        """Private method for internal use only."""
        pass
```

## Testing Guidelines

### Test Structure

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **Performance tests**: Test execution time and memory usage
- **Mock external dependencies**: Don't rely on external APIs in tests

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures and utilities
├── test_data_processor.py   # Data processing tests
├── test_model_builder.py    # Model building tests
├── test_trading_bot.py      # Trading bot tests
└── test_utils.py           # Utility function tests
```

### Writing Tests

```python
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.data_processor import DataProcessor
from tests.conftest import assert_dataframe_valid


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def test_fetch_data_success(self, data_processor, mock_data_fetcher):
        """Test successful data fetching."""
        data = data_processor.fetch_data('AAPL')
        
        assert_dataframe_valid(data)
        assert 'close' in data.columns
        assert len(data) > 0
    
    def test_fetch_data_invalid_symbol(self, data_processor):
        """Test error handling for invalid symbols."""
        with pytest.raises(ValueError):
            data_processor.fetch_data('INVALID_SYMBOL')
    
    @pytest.mark.parametrize("period,interval", [
        ("1y", "1d"),
        ("6mo", "1h"),
        ("1mo", "15m"),
    ])
    def test_different_timeframes(self, data_processor, period, interval):
        """Test data fetching with different timeframes."""
        data = data_processor.fetch_data('AAPL', period=period, interval=interval)
        assert_dataframe_valid(data)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_processor.py -v

# Run tests matching pattern
pytest tests/ -k "test_fetch_data"

# Run performance tests
pytest tests/ -m "performance"
```

## Documentation

### Documentation Requirements

- **Docstrings**: All public classes, functions, and methods
- **Type hints**: All function signatures
- **Examples**: Include usage examples for complex functionality
- **Architecture docs**: Update when adding new components

### Documentation Style

We use Google-style docstrings:

```python
def calculate_technical_indicators(
    self, 
    data: pd.DataFrame,
    indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate technical indicators for the given data.
    
    This method computes various technical indicators including trend,
    momentum, volatility, and volume-based indicators.
    
    Args:
        data: OHLCV market data with columns ['open', 'high', 'low', 'close', 'volume']
        indicators: Optional list of specific indicators to calculate.
                   If None, calculates all available indicators.
    
    Returns:
        DataFrame with original data plus calculated indicators
        
    Raises:
        ValueError: If data is missing required columns
        
    Examples:
        >>> processor = DataProcessor(config)
        >>> data = processor.fetch_data('AAPL')
        >>> indicators = processor.calculate_technical_indicators(data)
        >>> print(indicators.columns)
        ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 'macd', ...]
    """
```

## Pull Request Process

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout your-feature-branch
   git rebase develop
   ```

2. **Run all checks**:
   ```bash
   pre-commit run --all-files
   pytest tests/
   ```

3. **Update documentation** if needed

### Pull Request Template

Use this template for your pull request description:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code cleanup

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated existing tests if needed

## Documentation
- [ ] Updated docstrings
- [ ] Updated README if needed
- [ ] Updated architecture docs if needed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Commented hard-to-understand areas
- [ ] No breaking changes without proper documentation
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by at least one maintainer
3. **Testing** in various environments
4. **Documentation review** if applicable

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots/Logs**
If applicable, add screenshots or log output to help explain your problem.

**Environment:**
 - OS: [e.g. Ubuntu 20.04]
 - Python version: [e.g. 3.9.7]
 - Package version: [e.g. 1.0.0]

**Additional context**
Add any other context about the problem here.
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

## Development Guidelines

### Architecture Principles

1. **Modularity**: Keep components loosely coupled
2. **Testability**: Write testable code with clear interfaces
3. **Configuration**: Use configuration objects, avoid hard-coding
4. **Error handling**: Implement comprehensive error handling
5. **Logging**: Use structured logging throughout
6. **Performance**: Consider performance implications of changes

### Best Practices

1. **Keep functions small** and focused on single responsibility
2. **Use type hints** for better code documentation and IDE support
3. **Write defensive code** with proper input validation
4. **Follow DRY principle** - don't repeat yourself
5. **Use meaningful variable names** that explain purpose
6. **Comment complex logic** but prefer self-documenting code

### Performance Considerations

1. **Vectorized operations** when working with pandas/numpy
2. **Efficient data structures** for large datasets
3. **Memory management** - avoid memory leaks
4. **Caching strategies** for expensive operations
5. **Parallel processing** where appropriate

## Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

## Getting Help

- **Documentation**: Check the [docs/](docs/) directory
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Code examples**: Check the [examples/](examples/) directory

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for each release
- README.md contributors section
- GitHub contributor insights

Thank you for contributing to the Unified CNN-LSTM Trading Bot!