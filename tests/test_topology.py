import numpy as np

from gems.topology import fault_component_report, spatial_block_report


def test_connectivity_changes_diagonal_component_count():
    labels = np.zeros((5, 5), dtype=np.uint8)
    labels[1, 1] = 1
    labels[2, 2] = 1
    report8 = fault_component_report(labels, connectivity=8)
    report4 = fault_component_report(labels, connectivity=4)
    assert report8["component_count"] == 1
    assert report4["component_count"] == 2


def test_component_report_tracks_largest_share_and_bbox():
    labels = np.zeros((10, 10), dtype=np.uint8)
    labels[1:5, 2] = 1
    labels[7:9, 8] = 1
    report = fault_component_report(
        labels,
        connectivity=8,
        pixel_size_x_m=100,
        pixel_size_y_m=100,
        top_n=2,
    )
    assert report["fault_pixels"] == 6
    assert report["component_count"] == 2
    assert report["largest_component_pixels"] == 4
    assert report["largest_component_share"] == 4 / 6
    largest = report["top_components"][0]
    assert largest["bbox"]["height_pixels"] == 4
    assert largest["bbox"]["width_pixels"] == 1
    assert largest["bbox"]["height_m"] == 400.0


def test_component_report_honors_valid_mask():
    labels = np.zeros((6, 6), dtype=np.uint8)
    labels[1, 1] = 1
    labels[4, 4] = 1
    valid = np.ones((6, 6), dtype=bool)
    valid[4, 4] = False
    report = fault_component_report(labels, valid_mask=valid)
    assert report["fault_pixels"] == 1
    assert report["component_count"] == 1


def test_spatial_block_report_measures_label_sparsity():
    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[1, 1] = 1
    labels[6, 6] = 1
    report = spatial_block_report(labels, block_size=4)
    assert report["block_grid"] == [2, 2]
    assert report["valid_blocks"] == 4
    assert report["fault_positive_blocks"] == 2
    assert report["fault_positive_block_share"] == 0.5
