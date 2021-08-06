[![ci](https://github.com/rvbCMTS/PySkinDose/actions/workflows/ci.yml/badge.svg)](https://github.com/BwKodex/dicomimagetools/actions/workflows/python-package.yml) [![codecov](https://codecov.io/gh/BwKodex/dicomimagetools/branch/master/graph/badge.svg?token=2O32UO12V6)](https://codecov.io/gh/BwKodex/dicomimagetools)

# Dicom Image Tools 
The dicom image tools package was created for giving a framework handling DICOM image data. Adding functionality often used for programmatic image analysis.

The package is still in early development and more features will be added.

# Install Dicom Image Tools

Install using pipenv by running:

``
$ pipenv install dicom-image-tools
``

Install using pip by running:

``
$ pip install dicom-image-tools
``

Only **Python 3.7+** is supported.

See project wiki for more detailed documentation

# Usage

When you've installed the package you import it as any other package using 
 
```python 
import dicom_image_tools 
``` 

## Importing DICOM images
There are two functions for importing DICOM images, ``import_dicom_file(file: pathlib.Path)`` and ``import_dicom_from_folder(folder: pathlib.Path, recursively: bool = True)``.

The latter function has an optional input argument for specifying if the folder given should be searched for DICOM files recursively, default = ``True`` 

Both will return the image/-s in ``DicomStudy`` objects, the ``import_dicom_from_folder`` function returns a dictionary with the _Study Instance UID_ as the _key_ and the corresponding ``DicomStudy`` object as _value_.

You can then add additional files to the ``DicomStudy`` object through the ``DicomStudy.add_file`` which takes the file path as a ``pathlib.Path`` object as input.

The ``DicomStudy.Series`` is a list of all series belonging to the study that has been imported. Each ``DicomStudy.Series`` item is an object containing the image/image volume and metadata for each image. The image/image volume is accessed through the ``ImageVolume`` attribute of the ``DicomStudy.Series`` item, and the metadata in the ``CompleteMetadata`` attribute.

