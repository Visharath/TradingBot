from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="trading-bot",
    version="1.0.0",
    author="Visharath",
    author_email="visharath@example.com",
    description="A unified CNN-LSTM trading bot with advanced technical analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Visharath/TradingBot",
    project_urls={
        "Bug Tracker": "https://github.com/Visharath/TradingBot/issues",
        "Documentation": "https://github.com/Visharath/TradingBot/docs",
    },
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
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.1.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.15.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "trading-bot=src.unified_trading_bot:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)