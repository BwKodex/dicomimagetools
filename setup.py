from pathlib import Path
from setuptools import setup, find_packages

README = (Path(__file__).parent / 'README.md').read_text()

setup(
    name='dicom_image_tools',
    version='20.1.1',
    description='Python package for managing DICOM images from different modalities',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://dev.azure.com/bwkodex/DicomImageTools',
    author='Josef Lundman',
    author_email="josef@lundman.eu",
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pydicom>=1.3.0',
        'numpy>=1.18.1',
        'scikit-image>=0.16.0',
        'scipy>=1.4.1'
    ],
    zip_safe=False
)
