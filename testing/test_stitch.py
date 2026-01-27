import numpy as np
from backend.services.tiling import tile_image, stitch_tiles
from backend.services.data_loader import load_geotiff

image, _ = load_geotiff("christchurch_41.tif")

tiles = []
infos = []

for tile, info in tile_image(image):
    dummy_pred = np.ones((512, 512), dtype=np.float32)
    tiles.append(dummy_pred)
    infos.append(info)

full_mask = stitch_tiles(
    tiles,
    infos,
    full_shape=image.shape[:2]
)

print(full_mask.shape)
print(full_mask.max(), full_mask.min())
