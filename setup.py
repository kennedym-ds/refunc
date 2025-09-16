import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements/base.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setuptools.setup(
    name="refunc",
    version="0.1.0",
    author="Kennedy DS",
    author_email="your.email@example.com",
    description="A comprehensive ML utilities toolkit with advanced logging, performance monitoring, and file handling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kennedym-ds/refunc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=requirements,
    extras_require={
        "gpu": ["GPUtil>=1.4.0", "py3nvml>=0.2.0"],
        "dev": ["black>=22.0.0", "isort>=5.10.0", "flake8>=4.0.0", "mypy>=0.900", "pre-commit>=2.15.0"],
        "test": ["pytest>=6.2.0", "pytest-cov>=2.12.0", "pytest-mock>=3.6.0", "pytest-benchmark>=3.4.0"],
        "docs": ["sphinx>=4.0.0", "sphinx-rtd-theme>=1.0.0", "jupyter>=1.0.0", "nbsphinx>=0.8.0"],
    },
    keywords=["machine learning", "utilities", "logging", "performance", "decorators", "ml"],
    project_urls={
        "Bug Reports": "https://github.com/kennedym-ds/refunc/issues",
        "Source": "https://github.com/kennedym-ds/refunc",
        "Documentation": "https://github.com/kennedym-ds/refunc/blob/main/README.md",
    },
)
