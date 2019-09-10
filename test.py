from PIL import Image, ImageDraw

while True:
    path = input('Image path: ')
    image = Image.open(path)
    image_resize = image.resize((224, 224))
    draw = ImageDraw.Draw(image_resize)
    preditc = input('Predict: ')
    preditc = preditc.split(' ')
    for i in range(len(preditc)):
        preditc[i] = float(preditc[i])

    x, y, w, h = preditc[0] * 224, preditc[1] * 224, preditc[2] * 224, preditc[3] * 224
    draw.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], width=5)
    image_resize.show()
