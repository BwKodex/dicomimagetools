import pytest

from dicom_image_tools.helpers.geometry import Line


@pytest.mark.parametrize(
    "x0,y0,x1,y1",
    [
        ("0", 1, 1, 1),
        (1, "0", 1, 1),
        (1, 1, "0", 1),
        (1, 1, 1, "0"),
    ],
)
def test_Line_init_raises_TypeError_when_any_parameter_is_invalid_type(x0, y0, x1, y1):
    with pytest.raises(TypeError):
        _ = Line(x0=x0, y0=y0, x1=x1, y1=y1)
