from PIL import Image

numberOfThreads = 12

inputImagePath = "to4.jpg"
outputImagePath = "to3.jpg"


def brightness(point):
    return point / 10


image = Image.open(inputImagePath, 'r')

pixel_values = list(image.getdata())

# for r, g, b, a in pixel_values:
#     print(a)

# print(image.point())

out = image.point(lambda i: brightness(i))

# out.show()

out.save(outputImagePath)

print(image.format, image.size, image.mode)
