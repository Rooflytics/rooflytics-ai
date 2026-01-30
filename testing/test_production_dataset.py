from pathlib import Path
from training.dataset_production import RoofDatasetProduction

HERE = Path(__file__).parent

ds = RoofDatasetProduction(
    image_path=HERE / "christchurch_41.tif",
    mask_path=HERE / "christchurch_41_label.tif",
    max_tiles=100,
)

print("Dataset size:", len(ds))
x, y = ds[0]
print(x.shape, y.shape, y.min().item(), y.max().item())