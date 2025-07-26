"""
Setup script for the Unified CNN-LSTM Trading Bot.
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read version from __init__.py
init_file = Path(__file__).parent / "src" / "__init__.py"
version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read_text(), re.M)
if version_match:
    version = version_match.group(1)
else:
    raise RuntimeError("Unable to find version string.")

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text().splitlines() 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="unified-trading-bot",
    version=version,
    author="Visharath",
    author_email="your.email@example.com",
    description="Unified CNN-LSTM Trading Bot for Algorithmic Trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Visharath/TradingBot",
    project_urls={
        "Bug Reports": "https://github.com/Visharath/TradingBot/issues",
        "Source": "https://github.com/Visharath/TradingBot",
        "Documentation": "https://github.com/Visharath/TradingBot/docs",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.19.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipykernel>=6.0.0",
            "ipywidgets>=7.6.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0,<3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "trading-bot=unified_trading_bot.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "unified_trading_bot": [
            "config/*.json",
            "models/*.h5",
            "data/*.csv",
        ],
    },
    zip_safe=False,
    keywords=[
        "trading", "bot", "machine-learning", "cnn", "lstm", 
        "technical-analysis", "algorithmic-trading", "finance",
        "interactive-brokers", "tensorflow", "neural-network"
    ],
)