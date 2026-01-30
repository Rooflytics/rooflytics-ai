from backend.services.data_loader import load_geotiff
from backend.services.tiling import tile_image

image, _ = load_geotiff("christchurch_41.tif")

tiles = list(tile_image(image, tile_size=512))

print("Number of tiles:", len(tiles))
tile, info = tiles[0]
print("Tile shape:", tile.shape)
print("Tile info:", info)
