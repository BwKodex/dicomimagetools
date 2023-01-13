from pathlib import Path

from setuptools import find_packages, setup

README = (Path(__file__).parent / "README.md").read_text()

setup(
    name="dicom_image_tools",
    version="23.1.1",
    description="Python package for managing DICOM images from different modalities",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/BwKodex/dicomimagetools",
    author="Josef Lundman",
    author_email="josef@lundman.eu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pydicom>=2.3.1",
        "numpy>=1.24.0",
        "scikit-image>=0.17.2",
        "scipy>=1.9.3",
        "plotly>=5.11.0",
        "python-gdcm>=3.0.20"
    ],
    zip_safe=False,
)
