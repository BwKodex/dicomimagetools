from typing import Dict

from pydicom import FileDataset

from ..constants.SopClassUids import (
    RADIATION_DOSE_STRUCTURED_REPORT_SOP_CLASS_UIDS,
    SECONDARY_CAPTURE_SOP_CLASS_UIDS,
)


class DoseReport:
    def __init__(self):
        self.Rdsr: Dict[str, FileDataset] = {}
        self.SecondaryCapture: Dict[str, FileDataset] = {}

    def add_file(self, dataset: FileDataset):
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

            return

        if dataset.SOPClassUID in SECONDARY_CAPTURE_SOP_CLASS_UIDS:
            self.SecondaryCapture[
                (
                    f"{dataset.ContentDate if 'ContentDate' in dataset else ''}_{dataset.SOPInstanceUID}"
                    f"{dataset.ContentTime if 'ContentTime' in dataset else ''}_{dataset.SOPInstanceUID}"
                )
            ] = dataset
