python3 -c "
from PIL import Image
import numpy as np

mask = Image.open('landcover_ai/masks/M-33-20-D-c-4-2.tif')
arr = np.array(mask)
print('Shape:', arr.shape)
print('Unique values:', np.unique(arr.reshape(-1, arr.shape[2]), axis=0))
"