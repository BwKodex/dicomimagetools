from typing import Union
from pathlib import Path

from pydicom import FileDataset

from ..constants.SopClassUids import (
    RADIATION_DOSE_STRUCTURED_REPORT_SOP_CLASS_UIDS,
    SECONDARY_CAPTURE_SOP_CLASS_UIDS,
)


class DoseReport:
    def __init__(self):
        self.Rdsr: dict[str, FileDataset] = {}
        self.SecondaryCapture: dict[str, FileDataset] = {}
        self.RdsrFilePaths: list[Path] = []
        self.SecondaryCaptureFilePaths: list[Path] = []

    def add_file(self, file: Union[Path, str], dataset: FileDataset):
        if not isinstance(dataset, FileDataset):
            raise TypeError("The dataset is not a FileDataset")

        if (
            dataset.SOPClassUID not in RADIATION_DOSE_STRUCTURED_REPORT_SOP_CLASS_UIDS
            and dataset.SOPClassUID not in SECONDARY_CAPTURE_SOP_CLASS_UIDS
        ):
            raise ValueError("The supplied FileDataset is neither an RDSR nor a Secondary Capture")

        if dataset.SOPClassUID in RADIATION_DOSE_STRUCTURED_REPORT_SOP_CLASS_UIDS:
            self.Rdsr[
                (
                    f"{dataset.ContentDate if 'ContentDate' in dataset else ''}_{dataset.SOPInstanceUID}"
                    f"{dataset.ContentTime if 'ContentTime' in dataset else ''}_{dataset.SOPInstanceUID}"
                )
            ] = dataset
            self.RdsrFilePaths.append(file if isinstance(file, Path) else Path(file))

            return

        if dataset.SOPClassUID in SECONDARY_CAPTURE_SOP_CLASS_UIDS:
            self.SecondaryCapture[
                (
                    f"{dataset.ContentDate if 'ContentDate' in dataset else ''}_{dataset.SOPInstanceUID}"
                    f"{dataset.ContentTime if 'ContentTime' in dataset else ''}_{dataset.SOPInstanceUID}"
                )
            ] = dataset
            self.SecondaryCaptureFilePaths.append(file if isinstance(file, Path) else Path(file))
