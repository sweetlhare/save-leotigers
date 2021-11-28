# Репозиторий с решением ИИ-хакатона в Самаре 2021.

## Веб-приложение

Весь необходимый код для веб-приложения, развернутого на Streamlit находится в папке app_src.

Чтобы запустить приложение локально, нужно установить зависимости из requirments.txt и вызвать в консоли "streamlit run app.py"

## Инференс

python inference.py -p {path to folder} -t {task} -o {csv file with results}

-p (путь до папки с картинками)

-t (задача detect_and_classify / find_princess)

-o (файл с ответами)
