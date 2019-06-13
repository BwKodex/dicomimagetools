from dicom_image_tools.image_quality.mtf import mtf_bar_pattern
import pytest


def test_mtf_bar_pattern_raises_typeerror_max_hu():
    with pytest.raises(TypeError):
        mtf_bar_pattern(max_hu='Invalid type', bg=1.0, sd=[2.5, 2.4], sd_bg=0.5)


def test_mtf_bar_pattern_raises_typeerror_bg():
    with pytest.raises(TypeError):
        mtf_bar_pattern(max_hu=5.0, bg='Invalid type', sd=[2.5, 2.4], sd_bg=0.5)


def test_mtf_bar_pattern_raises_typeerror_sd_list():
    with pytest.raises(TypeError):
        mtf_bar_pattern(max_hu=5.0, bg=1.0, sd=2.5, sd_bg=0.5)


def test_mtf_bar_pattern_raises_typeerror_sd_list_element():
    with pytest.raises(TypeError):
        mtf_bar_pattern(max_hu=5.0, bg=1.0, sd=['Invalid', 'Type'], sd_bg=0.5)


def test_mtf_bar_pattern_raises_typeerror_sd_bg():
    with pytest.raises(TypeError):
        mtf_bar_pattern(max_hu=5.0, bg=1.0, sd=[2.5, 2.4], sd_bg='Invalid type')


def test_mtf_bar_pattern():
    expected = [1.3603, 1.3036]

    actual = mtf_bar_pattern(max_hu=5.0, bg=1.0, sd=[2.5, 2.4], sd_bg=0.5)

    assert [round(el, 4) for el in actual] == expected
