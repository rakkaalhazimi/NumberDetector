from PIL import Image


img = Image.open("drawing1.png")
img = img.resize((28, 28), Image.ANTIALIAS)
img.save("downsampled.png")