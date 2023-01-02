from dicom_image_tools.helpers.voxel_data import VoxelData


def test_voxel_data_correctly_calculates_the_pixel_area_in_x_y_plane():
    # Arrange
    voxel_data = VoxelData(x=5, y=6, z=0)
    expected = 30

    # Act
    actual = voxel_data.pixel_area()

    # Assert
    assert expected == actual


def test_voxel_data_correctly_calculates_the_voxel_volume():
    # Arrange
    voxel_data = VoxelData(x=5, y=6, z=2)
    expected = 60

    # Act
    actual = voxel_data.volume()

    # Assert
    assert expected == actual


def test_voxel_data_correctly_compares_different_instances_based_on_attribute_values():
    # Arrange
    voxel_data_1 = VoxelData(x=5, y=6, z=2)
    voxel_data_2 = VoxelData(x=5, y=6, z=2)
    voxel_data_3 = VoxelData(x=5, y=6, z=None)

    # Assert
    assert voxel_data_1 == voxel_data_2
    assert voxel_data_2 != voxel_data_3
