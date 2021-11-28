# Репозиторий с решением ИИ-хакатона в Самаре 2021.
## Команда – Московские Зайцы 🐰

Все необходимые файлы лежат на гугл диске: https://drive.google.com/drive/folders/1aeJ-JOyuVhp0P54rnsSq9VZtgPOLdq3o?usp=sharing

В коде они подгружаются автоматически

Ссылка на веб-приложение https://share.streamlit.io/sweetlhare/save-leotigers/main/app_src/app.py

Юпитер ноутбуки с подготовкой данных (но что-то из этого могло затеряться) и обучением моделей лежат в папке notebooks

В решении также были использованы данные с данного челленджа https://cvwc2019.github.io/challenge.html#

## Веб-приложение

Весь необходимый код для веб-приложения, развернутого на Streamlit находится в папке app_src.

Чтобы запустить приложение локально, нужно установить зависимости из requirments.txt и вызвать в консоли 

streamlit run app.py

## Инференс

python inference.py -p {path to folder} -t {task} -o {csv file with results}

-p (путь до папки с картинками)

-t (задача detect_and_classify / find_princess)

-o (файл с ответами)
