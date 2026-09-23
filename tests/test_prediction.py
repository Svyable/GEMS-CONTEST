import numpy as np
import rasterio
from rasterio.transform import from_origin

from gems.prediction import write_prediction_like_template
from gems.submission import validate_submission


def _template(path):
    data = np.zeros((1, 5, 6), dtype=np.float32)
    data[:, 0, :] = np.nan
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=6,
        height=5,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=from_origin(300000, 4400000, 100, 100),
        nodata=np.nan,
    ) as dst:
        dst.write(data)


def test_write_prediction_matches_template_and_validates(tmp_path):
    template = tmp_path / "template.tif"
    output = tmp_path / "prediction.tif"
    _template(template)

    prediction = np.full((5, 6), 0.4, dtype=np.float32)
    write_prediction_like_template(
        prediction,
        template_path=template,
        output_path=output,
    )

    report = validate_submission(output, template)
    assert report.ok, report.errors
    with rasterio.open(output) as src:
        values = src.read(1)
        assert np.isnan(values[0]).all()
        assert np.allclose(values[1:], 0.4)
