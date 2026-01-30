import numpy as np
from backend.services.preprocessing import detect_shadows

dummy = np.random.rand(512, 512, 3)
shadow = detect_shadows(dummy)

print(shadow.shape)
print(shadow.dtype)
print("Shadow %:", shadow.mean() * 100)
