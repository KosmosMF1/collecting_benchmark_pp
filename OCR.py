import numpy as np
import pandas as pd
import cv2
import os
import gc
import json
import glob
import re
import PIL
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from collections import Counter
from matplotlib.path import Path
from torchvision import models
from torchvision import transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import CharErrorRate as CER
from IPython.display import Image as Img


def decode(pred, alphabet):
    pred = pred.permute(1, 0, 2).cpu().data.numpy()
    outputs = []
    for i in range(len(pred)):
        outputs.append(pred_to_string(pred[i], alphabet))
    return outputs


def pred_to_string(pred, alphabet):
    seq = []
    for i in range(len(pred)):
        label = np.argmax(pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = ''.join([alphabet[c] for c in out])
    return out


def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)


# Чтобы без проблем реализовывать json. Без него есть нюансы
class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


PATH = 'C:/Users/Andrew/PycharmProjects/pythonProject/autoriaNumberplateOcrRu'
OCR_MODEL_PATH = 'C:/Users/Andrew/PycharmProjects/pythonProject/models/model-7-0.9156.ckpt'
ALPHABET = '0123456789ABEKMHOPCTYX'
TRAIN_SIZE = 0.9
BATCH_SIZE_OCR = 16

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#print(device)
#print(torch.cuda.is_available())

def get_annot(file, dir):
    return load_json(PATH + f'/{dir}/ann/' + file[:-3] + 'json')['description']


train_labels = pd.DataFrame([[f, get_annot(f, 'train')] for f in os.listdir(PATH + '/train/img')], columns=['filename', 'label']).to_dict('records')
val_labels = pd.DataFrame([[f, get_annot(f, 'val')] for f in os.listdir(PATH + '/val/img')], columns=['filename', 'label']).to_dict('records')
#print(train_labels[:10])


class OCRDataset(Dataset):
    def __init__(self, marks, img_folder, tokenizer, transforms=None):
        self.img_paths = []
        self.texts = []
        for item in marks:
            self.img_paths.append(os.path.join(PATH + f'/{img_folder}/img/', item['filename']))
            self.texts.append(item['label'])

        self.enc_texts = tokenizer.encode(self.texts)
        self.img_folder = PATH + f'/{img_folder}/img/'
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        #        print(img_path)
        text = self.texts[idx]
        enc_text = torch.LongTensor(self.enc_texts[idx])
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image)

        return image, text, enc_text

    def __len__(self):
        return len(self.texts)


class Resize(object):
    def __init__(self, size=(250, 50)):
        self.size = size

    def __call__(self, img):
        w_from, h_from = img.shape[1], img.shape[0]
        w_to, h_to = self.size

        # Сделаем разную интерполяцию при увеличении и уменьшении
        # Если увеличиваем картинку, меняем интерполяцию
        interpolation = cv2.INTER_AREA
        if w_to > w_from:
            interpolation = cv2.INTER_CUBIC

        img = cv2.resize(img, dsize=self.size, interpolation=interpolation)
        return img


class Normalize:
    def __call__(self, img):
        img = img.astype(np.float32) / 255
        return img


def collate_fn(batch):
    images, texts, enc_texts = zip(*batch)
    images = torch.stack(images, 0)
    text_lens = torch.LongTensor([len(text) for text in texts])
    enc_pad_texts = pad_sequence(enc_texts, batch_first=True, padding_value=0)
    return images, texts, enc_pad_texts, text_lens

OOV_TOKEN = '<OOV>'
CTC_BLANK = '<BLANK>'


def get_char_map(alphabet):
    char_map = {value: idx + 2 for (idx, value) in enumerate(alphabet)}
    char_map[CTC_BLANK] = 0
    char_map[OOV_TOKEN] = 1
    return char_map


class Tokenizer:
    def __init__(self, alphabet):
        self.char_map = get_char_map(alphabet)
        self.rev_char_map = {val: key for key, val in self.char_map.items()}

    def encode(self, word_list):
        enc_words = []
        for word in word_list:
            enc_words.append(
                [self.char_map[char] if char in self.char_map
                 else self.char_map[OOV_TOKEN]
                 for char in word]
            )
        return enc_words

    def get_num_chars(self):
        return len(self.char_map)

    def decode(self, enc_word_list):
        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            for idx, char_enc in enumerate(word):
                # skip if blank symbol, oov token or repeated characters
                if (
                    char_enc != self.char_map[OOV_TOKEN]
                    and char_enc != self.char_map[CTC_BLANK]
                    # idx > 0 to avoid selecting [-1] item
                    and not (idx > 0 and char_enc == word[idx - 1])
                ):
                    word_chars += self.rev_char_map[char_enc]
            dec_words.append(word_chars)
        return dec_words

ocr_transforms = transforms.Compose([
    Resize(size=(250, 50)),
    Normalize(),
    transforms.ToTensor()
])

tokenizer = Tokenizer(ALPHABET)

train_ocr_dataset = OCRDataset(
    marks=train_labels,
    img_folder='train',
    tokenizer=tokenizer,
    transforms=ocr_transforms
)
val_ocr_dataset = OCRDataset(
    marks=val_labels,
    img_folder='val',
    tokenizer=tokenizer,
    transforms=ocr_transforms
)

train_loader = DataLoader(
    train_ocr_dataset,
    batch_size=BATCH_SIZE_OCR,
    drop_last=True,
    num_workers=0,  # Изменение здесь
    collate_fn=collate_fn,
    timeout=0,
    shuffle=True
)

val_loader = DataLoader(
    val_ocr_dataset,
    batch_size=BATCH_SIZE_OCR,
    drop_last=False,
    num_workers=0,  # Изменение здесь
    collate_fn=collate_fn,
    timeout=0,
)

gc.collect()

#print(train_ocr_dataset[0])

img_tensor = train_ocr_dataset[0][0]  # Это тензор изображения
plt.imshow(img_tensor.permute(1, 2, 0))  # Меняем порядок на H x W x C
plt.title(train_ocr_dataset[0][1])  # Заголовок с номером
plt.axis('off')  # Отключаем оси
#plt.show()

def get_resnet34_backbone():
    m = models.resnet34(pretrained=True)
    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    blocks = [input_conv, m.bn1, m.relu,
              m.maxpool, m.layer1, m.layer2, m.layer3]
    return nn.Sequential(*blocks)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class CRNN(nn.Module):
    def __init__(
        self, number_class_symbols, time_feature_count=256, lstm_hidden=256,
        lstm_len=2,
    ):
        super().__init__()
        self.feature_extractor = get_resnet34_backbone()
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (time_feature_count, time_feature_count))
        self.bilstm = BiLSTM(time_feature_count, lstm_hidden, lstm_len)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, number_class_symbols)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)
        return x

model = CRNN(number_class_symbols=tokenizer.get_num_chars())
model.to(device)

criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                              weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode='max', factor=0.5, patience=15)

class AverageMeter:
    """Вычисляет и хранит среднее значение"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_accuracy(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        scores.append(true == pred)
    avg_score = np.mean(scores)
    return avg_score


def predict(images, model, tokenizer, device):
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    pred = torch.argmax(output.detach().cpu(), -1).permute(1, 0).numpy()
    text_preds = tokenizer.decode(pred)
    return text_preds


def train_loop(data_loader, model, criterion, optimizer, epoch):
    loss_avg = AverageMeter()
    model.train()
    # Устанавливаем tqdm для отображения прогресса
    with tqdm(total=len(data_loader), desc=f'Training Epoch {epoch+1}', leave=True) as pbar:
        for images, texts, enc_pad_texts, text_lens in data_loader:
            model.zero_grad()
            images = images.to(device)
            batch_size = len(texts)
            output = model(images)
            output_lengths = torch.full(
                size=(output.size(1),),
                fill_value=output.size(0),
                dtype=torch.long
            )
            loss = criterion(output, enc_pad_texts, output_lengths, text_lens)
            loss_avg.update(loss.item(), batch_size)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()

            # Обновляем прогресс
            pbar.set_postfix(loss=loss_avg.avg)
            pbar.update(1)

    print(f'Epoch {epoch+1}, Loss: {loss_avg.avg:.5f}')
    return loss_avg.avg


def val_loop(data_loader, model, tokenizer, device):
    acc_avg = AverageMeter()
    # Устанавливаем tqdm для отображения прогресса
    with tqdm(total=len(data_loader), desc='Validation', leave=True) as pbar:
        for images, texts, _, _ in data_loader:
            batch_size = len(texts)
            text_preds = predict(images, model, tokenizer, device)
            acc_avg.update(get_accuracy(texts, text_preds), batch_size)

            # Обновляем прогресс
            pbar.update(1)

    print(f'Validation, acc: {acc_avg.avg:.4f}\n')
    return acc_avg.avg

'''def train(model, optimizer, scheduler, train_loader, val_loader, epochs=10):
    best_acc = -np.inf
    os.makedirs('models', exist_ok=True)
    acc_avg = val_loop(val_loader, model, tokenizer, device)
    for epoch in range(epochs):
        print(f'Epoch {epoch} started')
        loss_avg = train_loop(train_loader, model, criterion, optimizer, epoch)
        acc_avg = val_loop(val_loader, model, tokenizer, device)
        scheduler.step(acc_avg)
        if acc_avg > best_acc:
            best_acc = acc_avg
            model_save_path = os.path.join(
                'models', f'model-{epoch}-{acc_avg:.4f}.ckpt')
            torch.save(model.state_dict(), model_save_path)
            print('Model weights saved')


train(model, optimizer, scheduler, train_loader, val_loader, epochs=10)'''


class InferenceTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images):
        transformed_images = []
        for image in images:
            # Применяем преобразования
            transformed_image = self.transforms(image)
            transformed_images.append(transformed_image)

        # Стек всех преобразованных изображений в один тензор
        transformed_tensor = torch.stack(transformed_images)
        return transformed_tensor


class Predictor:
    def __init__(self, model_path, device='cuda'):
        self.tokenizer = Tokenizer(ALPHABET)
        self.device = torch.device(device)

        # Загружаем модель
        self.model = CRNN(number_class_symbols=self.tokenizer.get_num_chars())
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

        self.transforms = InferenceTransform(ocr_transforms)

    def __call__(self, images):
        if isinstance(images, np.ndarray):
            images = [images]  # Преобразуем в список, если это одно изображение
            single_image = True
        elif isinstance(images, (list, tuple)):
            single_image = False
        else:
            raise TypeError(f"Input must be np.ndarray, list, or tuple, found {type(images)}.")

        # Применяем трансформации
        images = self.transforms(images)
        images = images.to(self.device)  # Переносим на нужное устройство

        # Получаем предсказания
        predictions = self.predict(images)

        if single_image:
            return predictions[0]  # Возвращаем единственное предсказание
        return predictions

    def predict(self, images):
        self.model.eval()  # Устанавливаем модель в режим оценки
        with torch.no_grad():
            output = self.model(images)  # Получаем предсказания модели

        pred = torch.argmax(output.detach().cpu(), dim=-1).permute(1, 0).numpy()
        text_predictions = self.tokenizer.decode(pred)  # Декодируем предсказания
        return text_predictions