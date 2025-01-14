from setuptools import setup, find_packages

setup(
    name="torchgalayers",  # Replace with your package name
    version="0.1.1",  # Package version
    author="Alberto Maria Pepe",
    author_email="ap2219@cam.ac.uk",
    description="clifford algebra neural layers in pytorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/albertomariapepe/Torch-GA",
    packages=find_packages(),  # Automatically find sub-packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.10.0",  # Replace with the minimum required version of PyTorch
        "numpy>=1.21.0",
    ],
)