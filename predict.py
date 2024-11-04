import cv2
import matplotlib.pyplot as plt
from OCR import Predictor


# Функция для открытия и конвертации изображения
def open_img(img_path):
    carplate_img = cv2.imread(img_path)
    if carplate_img is None:
        raise ValueError(f"Не удалось загрузить изображение: {img_path}")
    carplate_img = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)
    return carplate_img


# Функция для обнаружения и извлечения номерных знаков
def carplate_extract(image, carplate_haar_cascade):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Преобразование в серый цвет для работы каскада
    carplate_rects = carplate_haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    carplates = []  # Список для хранения найденных номерных знаков

    for x, y, w, h in carplate_rects:
        carplate_img = image[y:y + h, x:x + w]
        carplates.append(carplate_img)  # Добавляем извлечённый номерной знак в список

    return carplates  # Возвращаем список всех найденных номерных знаков


# Функция для увеличения изображения
def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


# Функция для распознавания текста на изображении номерного знака
def recognize_license_plate(plate_img, model_path):
    # Преобразование изображения номерного знака в формат, подходящий для модели
    gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
    rgb_plate_img = cv2.cvtColor(gray_plate_img, cv2.COLOR_GRAY2RGB)

    # Инициализация и использование предсказателя
    predictor = Predictor(model_path=model_path)
    prediction = predictor(rgb_plate_img)
    return prediction


# Основная функция, объединяющая оба этапа
def detect_and_recognize_plate(image_path, model_path, cascade_path):
    # Открываем изображение
    carplate_img_rgb = open_img(image_path)

    # Инициализируем каскад Хаара для обнаружения номерного знака
    carplate_haar_cascade = cv2.CascadeClassifier(cascade_path)

    # Извлечение номерных знаков
    carplates = carplate_extract(carplate_img_rgb, carplate_haar_cascade)

    if not carplates:
        print(f"Номерные знаки не найдены для {image_path}.")
        return

    # Распознаем текст на всех извлечённых номерных знаках
    for idx, carplate_extract_img in enumerate(carplates):
        # Увеличиваем изображение для лучшего распознавания
        carplate_extract_img = enlarge_img(carplate_extract_img, 150)

        # Распознаем текст на извлечённом номерном знаке
        license_text = recognize_license_plate(carplate_extract_img, model_path)

        # Отображение результата
        plt.axis('off')
        plt.imshow(carplate_extract_img)
        plt.title(f"Номерной знак {idx + 1}")
        plt.show()
        print(f"Распознанный текст для номерного знака {idx + 1}: {license_text}")


# Пример использования
if __name__ == '__main__':
    image_paths = [
        'C:/Users/Andrew/PycharmProjects/pythonProject/CarImages/A777AA177.jpg',
        'C:/Users/Andrew/PycharmProjects/pythonProject/CarImages/H757YC37.jpg',
        'C:/Users/Andrew/PycharmProjects/pythonProject/CarImages/M611OC32, E611CO32.png',
        'C:/Users/Andrew/PycharmProjects/pythonProject/CarImages/X010XX71, A010AA71.jpg',
        'C:/Users/Andrew/PycharmProjects/pythonProject/CarImages/Y003KK190, X248YM150, A082MP97.jpg'
    ]
    model_path = 'C:/Users/Andrew/PycharmProjects/pythonProject/models/model-9-0.9943.ckpt'
    cascade_path = 'C:/Users/Andrew/PycharmProjects/pythonProject/haarcascade_russian_plate_number.xml'

    for img_path in image_paths:
        detect_and_recognize_plate(img_path, model_path, cascade_path)
