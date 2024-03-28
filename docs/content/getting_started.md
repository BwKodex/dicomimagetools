# Getting started

## Importing DICOM images

There are two functions for importing DICOM images, ``import_dicom_file(file)`` and ``import_dicom_from_folder(folder, recursively=True)``.

The latter function has an optional input argument for specifying if the folder given should be searched for DICOM files recursively, default = ``True`` 

The input arguments ``file`` and ``folder`` in the ``import_dicom_file`` and ``import_dicom_from_folder``, respectively, should be given as either a ``pathlib.Path`` object or a string.

Both import functions will return the image/-s in ``DicomStudy`` objects, the ``import_dicom_from_folder`` function returns a dictionary with the _Study Instance UID_ as the _key_ and the corresponding ``DicomStudy`` object as _value_.

You can then add additional files to the ``DicomStudy`` object through the ``DicomStudy.add_file`` which takes the file path as a ``pathlib.Path`` object as input.

The ``DicomStudy.Series`` is a list of all series belonging to the study that has been imported. Each ``DicomStudy.Series`` item is an object containing the image/image volume and metadata for each image. The image/image volume is accessed through the ``ImageVolume`` attribute of the ``DicomStudy.Series`` item, and the metadata in the ``CompleteMetadata`` attribute.

To not overload the memory when importing a large number of DICOM files, the image/image volume is not immediately imported into the ``ImageVolume``. When you want to access the images you first have to call the ``import_image`` or ``import_image_volume``, which one depends on whether the series is a CT or projection image series, method on the series you want to access.

So a short script importing from a directory containing CT images and then importing the image volume would look something like this

````python
import dicom_image_tools as dit

dicom_study_dict = dit.import_dicom_from_folder(folder="/path/to/directory/containing/ct-images")

dicom_studies_imported = list(dicom_study_dict.keys())

print(
    f"Imported images from {len(dicom_studies_imported)}" 
    f"{'study' if len(dicom_studies_imported) == 1 else 'studies'}"
)
print(
    "The StudyInstanceUIDs of the imported "
    f"{'study is' if len(dicom_studies_imported) == 1 else 'studies are'}"
)
for study_key in dicom_studies_imported:
    print(f"\t{study_key}")

first_dicom_study = dicom_study_dict[dicom_studies_imported[0]]
first_dicom_study.Series[0].import_image_volume()
````

## Accession DICOM metadata

The DICOM metadata of the images in the DICOM study is stored in the ``CompleteMetadata`` attribute of the ``DicomStudy.Series`` it belongs to. The ``CompleteMetadata`` is a list of DICOM datasets (see the [pydicom documentation on datasets](https://pydicom.github.io/pydicom/dev/reference/generated/pydicom.dataset.Dataset.html)) stored in the same order as the images in the series. This means that the metadata for the first image in the ``ImageVolume`` of the imported image volume in the ``first_dicom_study`` in the example above would be accessed through

```python
first_image_metadata = first_dicom_study.Series[0].CompleteMetadata[0]
```
