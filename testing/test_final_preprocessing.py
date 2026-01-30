import numpy as np
from backend.services.preprocessing import preprocess_tile

dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

tile_norm, shadow = preprocess_tile(dummy)

print(tile_norm.shape, tile_norm.dtype)
print(shadow.shape, shadow.dtype)
print("Shadow %:", shadow.mean() * 100)