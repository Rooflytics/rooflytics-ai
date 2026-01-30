import numpy as np
from backend.services.preprocessing import normalize_tile

dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

img_norm = normalize_tile(dummy, method="imagenet")
print(img_norm.mean(axis=(0, 1)))
print(img_norm.std(axis=(0, 1)))

img_std = normalize_tile(dummy, method="per_image")
print(img_std.mean(axis=(0, 1)))
print(img_std.std(axis=(0, 1)))
