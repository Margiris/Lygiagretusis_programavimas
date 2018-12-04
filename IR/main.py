# Pillow >= 1.0 no longer supports “import Image”. Please use “from PIL import Image” instead.
# Pillow >= 2.1.0 no longer supports “import _imaging”. Please use “from PIL.Image import core as _imaging” instead.

from PIL import Image

import gui

numberOfThreads = 12
outputImagePath = "aaa.png"

# inputImagePath = "test1.png"
inputImagePath = "to2.jpg"


def brightness(point):
    return point * 10


image = Image.open(inputImagePath, 'r')

pixel_values = list(image.getdata())

# for r, g, b, a in pixel_values:
#     print(a)

# print(image.point())

out = image.point(lambda i: brightness(i))

# out.show()

out.save(outputImagePath)

print(image.format, image.size, image.mode)
