from PIL import Image, ImageDraw

while True:
    path = input('Image path: ')
    image = Image.open(path)
    image_resize = image.resize((224, 224))
    draw = ImageDraw.Draw(image_resize)
    predict = input('Predict: ')
    predict = predict.split(' ')
    for i in range(len(predict)):
        predict[i] = float(predict[i])

    x, y, w, h = predict[0] * 224, predict[1] * 224, predict[2] * 224, predict[3] * 224
    draw.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], width=5)
    image_resize.show()
