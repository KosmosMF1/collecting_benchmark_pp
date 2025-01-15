import asyncio
import time
import OCR

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message, ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton
from test import process_multiple_images, detect_and_recognize_plate, MODEL_PATH, CASCADE_PATH

from config import token

bot = Bot(token=token)
dp = Dispatcher()

# Создаем клавиатуру с кнопками
start_button = KeyboardButton(text="Запустить бота еще раз")
team_button = KeyboardButton(text="Команда разработчиков")

@dp.message(CommandStart())
async def start_message(message: Message):
    await message.answer('Привет, добро пожаловать в чат-бот по распознаванию номеров автомобилей\n'
                         '\n'
                         'Загрузите фотографию с номером автомобиля.', reply_markup=ReplyKeyboardRemove())

@dp.message(F.photo)
async def photo_handler(message: Message) -> None:
    file_name = f"photos/{message.photo[-1].file_id}.jpg"
    await message.bot.download(file=message.photo[-1].file_id, destination=file_name)
    await message.answer('Ожидайте результата...')
    result = detect_and_recognize_plate(file_name, MODEL_PATH, CASCADE_PATH)
    await message.answer(f"Распознанный номер: {result}")

    keyboard = ReplyKeyboardMarkup(
        keyboard=[[start_button, team_button]],
        resize_keyboard=True
    )
    await message.answer('Выберите действие:', reply_markup=keyboard)

@dp.message()
async def other_message(message: Message):
    if message.text == "Запустить бота еще раз":
        await start_message(message)
    if message.text == "Команда разработчиков":
        await message.answer('Арсланов Андрей - разработчик\n'
                             'Марков Артем - аналитик\n'
                             'Филиппов Михаил - тимлид')

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Выход из программы')