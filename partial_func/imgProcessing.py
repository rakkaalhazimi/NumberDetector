import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

filename = r"downsampled.png"

img = Image.open(filename).convert("L")
# img.save("converted.png")

img = np.asarray(img)
img = img.reshape(28, 28)
print(img.shape)
plt.imshow(img)
plt.show()