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
        'pydicom>=2.1.1',
        'numpy>=1.19.0, <=1.19.2',
        'scikit-image>=0.17.2',
        'scipy>=1.5.4'
    ],
    zip_safe=False
)
