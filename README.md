**ML-сервис, который по ulr (которая, предположительно, ведет на один из множества сайтов, на которых выложены книги на русском языке) пытается определить, верно ли, что это страница с непосредственно текстом - или это какая-то промежуточная страница, например, где описывается книга или какой-то раздел сайта.**

Он должен был стать частью проекта, который потом страницы с текстом книг скачивает, выделяет сам текст и индексирует все это, чтобы, в итоге, создать сервис, который эффективно ищет цитаты именно по литературному первоисточнику (суть проекта была в том, чтобы отличать фейковые цитаты авторов из прошлого от нефейковых).

**Модель обучения - Random Forest (sklearn) + перевод в .onnx формат для хранения. В модели зашиты метаданные (дата ее создания, название эксперимента и хэш коммита в гите, на основе которого производилось создание модели).**

Приложение app.py создает веб-сервис для применения модели (хост, порт и адрес модели - параметры). На запрос типа POST с json-документом, в котором есть поле "url", модель возвращает предсказание. На запрос /metadata возвращаются метаданные. 

Запросы:

**/metadata** : метаданные

**/forward [тип POST]** : требуется передать json-документ с полем url, результатом будет json-документ с полем 'answer' и его значением 0 или 1 - предсказанием модели

**/forward_batch [тип POST]** : требуется передать .csv файл с колонкой url; результатом будет json-документ с полем 'answer' и массивом со значениями {0, 1, None} - последнее в случае, если модель не смогла обработать url

**/evaluate [тип POST]** : аналогично, но в .csv файле должно быть поле is_text, а в ответ добавляется поле metrics c метриками по тем url, по которым удалось получить предсказание

Также представлен код самого обучения и выделения признаков (на вход подается только url, признаки строятся по нему с помощью BeautifulSoup). И тесты. 

**Для удобства тестирования последних двух видов запроса приложен файл test.csv, который можно отправлять на сервер. Датасет можно скачать вот [здесь](https://drive.google.com/file/d/1jGdyroDIz3iLT_pbw86IfVwhIW4vHMRs/view?usp=sharing)**

**При сборке контейнера путь к обученной модели должен лежать в переменной среды MODEL_PATH**

**Готовый образ можно взять [здесь](https://hub.docker.com/r/masha239/texts)**
