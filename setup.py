from setuptools import setup, find_packages

setup(
    name='dicom_image_tools',
    version='0.0.1',
    description='Python package for managing DICOM images from different modalities',
    url='',
    author='Josef Lundman',
    author_email="josef.lundman@gmail.com",
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.7'
    ],
    packages=find_packages(),
    install_requires=[
        'pydicom>=1.2.2',
        'numpy>=1.16.3',
        'scikit-image>=0.15.0'
    ],
    zip_safe=False
)
