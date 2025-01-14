from setuptools import setup, find_packages

setup(
    name="annalog",
    version="0.3.5",
    packages=find_packages(),  # Automatically find packages
    install_requires=[
        "partialsmiles @ git+https://github.com/baoilleach/partialsmiles.git"
    ],
    description="ANNalog -- a seq2seq model which could generate similar molecules",
    author="Wei Dai",
    author_email="bty415@qmul.ac.uk",
    license="MIT",
    url="https://github.com/DVNecromancer/ANNalog/tree/main",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
