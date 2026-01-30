from backend.services.data_loader import load_geotiff

img, meta = load_geotiff("christchurch_41.tif")

print(img.shape)
print(meta["crs"])
print(meta["transform"])
