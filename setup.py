from setuptools import setup, find_packages

setup(
    name="remove_confidential",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.12.0",
    ],
    entry_points={
        "console_scripts": [
            "remove-confidential=remove_confidential.remove_confidential:main",
        ],
    },
)