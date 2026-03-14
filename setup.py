from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent
README_PATH = HERE / "README.md"
README = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""

setup(
    name="annalog",
    version="1.0.4",
    packages=find_packages(),
    include_package_data=True,
    package_data={"annalog": ["ckpt_and_vocab/*"]},
    description="ANNalog — a SMILES-to-SMILES seq2seq model for medchem analogue generation",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Wei Dai",
    author_email="bty415@qmul.ac.uk",
    license="MIT",
    url="https://github.com/DVNecromancer/ANNalog",
    project_urls={
        "Source": "https://github.com/DVNecromancer/ANNalog",
        "Issues": "https://github.com/DVNecromancer/ANNalog/issues",
    },
    install_requires=[
        "partialsmiles>=2.0",
        "chembl-gen-check==0.1.2",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "annalog-generate=annalog.cli:cli",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
