import cv2
import os
import matplotlib.pyplot as plt
from OCR import Predictor

# Загрузка каскада Хаара и модели
CASCADE_PATH = "C:/Users/Andrew/PycharmProjects/pythonProject/haarcascade_russian_plate_number.xml"
MODEL_PATH = "C:/Users/Andrew/PycharmProjects/pythonProject/models/model-8-0.9971.ckpt"
cascade = cv2.CascadeClassifier(CASCADE_PATH)
predictor = Predictor(model_path=MODEL_PATH)

def preprocess_image(image):
    """
    Предобработка изображения: преобразование в градации серого.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def detect_and_recognize_plate(image_path, model_path, cascade_path):
    # Открываем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: изображение {image_path} не загружено.")
        return

    # Предобработка изображения
    gray = preprocess_image(image)

    # Инициализируем каскад Хаара для обнаружения номерного знака
    carplate_cascade = cv2.CascadeClassifier(cascade_path)

    # Извлечение номерных знаков
    carplates = carplate_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 15), maxSize=(300, 100)
    )

    if len(carplates) == 0:
        print(f"Номерные знаки не найдены для {image_path}.")
        return

    image_name = os.path.basename(image_path)

    # Распознаем текст на всех извлечённых номерных знаках
    for idx, (x, y, w, h) in enumerate(carplates):
        # Извлечение и увеличение номерного знака
        carplate_img = image[y:y + h, x:x + w]
        carplate_img = cv2.resize(carplate_img, (250, 50))

        # Распознаем текст на извлечённом номерном знаке
        license_text = predictor(carplate_img)

        # Отображение результата
        plt.figure(figsize=(8, 4))
        plt.axis('off')
        plt.imshow(cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB))

        # Добавление текста с увеличенным отступом
        plt.text(0.5, -0.15, f"Распознанный текст: {license_text}", fontsize=12, ha='center',
                 transform=plt.gca().transAxes)

        # Заголовок
        plt.title(f"Номерной знак {idx + 1}")

        # Показ изображения с текстом
        plt.show()

        print(f"{image_name}: Распознанный текст на фото: {license_text}")

        return license_text

def process_multiple_images(image_paths, model_path, cascade_path):
    """
    Обрабатывает несколько изображений.
    """
    for image_path in image_paths:
         return detect_and_recognize_plate(image_path, model_path, cascade_path)
