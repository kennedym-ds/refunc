import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="refunc",
    version="0.0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive ML utilities toolkit.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kennedym-ds/refunc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
