from dicom_image_tools.dicom_handlers.ct import CtSeries


def test_ct_series_creation():
    expected = dict(
        type=CtSeries,
        series_instance_uid='SomeSeriesInstanceUid'
    )
    ct_series_obj = CtSeries(series_instance_uid=expected.get('series_instance_uid'))

    assert isinstance(ct_series_obj, expected.get('type'))
    assert ct_series_obj.SeriesInstanceUid == expected.get('series_instance_uid')
