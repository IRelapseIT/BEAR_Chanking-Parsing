# BEAR PDF OCR

Автономный экспорт актуального PDF-парсера BEAR. На входе локальный PDF с текстовым слоем, сканами или смешанным содержимым. На выходе JSON с текстом, таблицами, изображениями, координатами источников и причинами ручной проверки. Основной сценарий для сканов — страницы внутри PDF.

## Стек

| Компонент | Назначение |
|---|---|
| Python 3.12 | Основной процесс, CLI, модели данных и запуск изолированных OCR-процессов |
| pdfplumber 0.11.10 | Текстовый слой PDF, геометрия слов, таблицы с линиями |
| pypdfium2 5.14.0 | Растеризация страниц и областей |
| Pillow 13.0.0 | Подготовка изображений и проверка пустых областей |
| Tesseract 5.5.2, `eng+rus` | Основной OCR на CPU. Для страниц возвращает TSV с координатами и confidence |
| PaddleOCR 3.7.0, PaddleX 3.7.2, PaddlePaddle 3.3.1 | Дополнительный CPU-процесс. `PP-OCRv5_mobile_det` находит строки, `eslav_PP-OCRv5_mobile_rec` читает их для независимой сверки |
| PP-DocLayout_plus-L | Дополнительное обнаружение области таблицы по проблемной странице |
| Docling 2.148.0 | Дополнительный постраничный резерв для порядка чтения и структуры таблиц. OCR в этой конфигурации также выполняет Tesseract |
| psutil 7.2.2 | Контроль памяти дерева процесса Paddle |

## Архитектура

```text
PDF → pdf_adapter → текстовый слой + координаты + таблицы
        ├─ скан или область без текста → PDFium → Tesseract → TSV
        │                                         └─ тот же растр → PaddleOCR → сверка
        ├─ встроенное изображение → OCR надписи + привязка подписи к источнику
        └─ подпись таблицы без сетки → детектор области, если включён
    → pdf_fallback → проверка проблемной страницы, при наличии Docling
    → ParsedDocument → JSON
```

1. `pdf_adapter.py` обходит страницы. Для текстового PDF сохраняет слова и координаты, собирает строки и использует расстояния между колонками для порядка чтения. Таблицы с рамками проверяются на сохранность извлечённого текста.
2. `pdf_table_geometry.py` восстанавливает простую таблицу без рамок по регулярной сетке слов под подписью. Ячейки содержат координаты и индексы строки и колонки. Статус страницы остаётся `needs_review`.
3. OCR основной страницы использует растр PDFium с DPI 216 по умолчанию. `ocr_verification.py` переводит координаты OCR в координаты PDF и сопоставляет строки по пересечению областей. Отчёт содержит `agreement`, `disagreement`, `insufficient` или `not_run`, оба чтения и причины расхождений. Символы знака, числа и единицы участвуют в сравнении.
4. `visual_context.py` связывает изображение с ближайшей подписью по странице и геометрии. OCR надписи находится в `metadata.image_ocr`. Эмблема без надписи может дать `no_text_detected`; идентификация бренда требует отдельного метода.
5. `pdf_fallback.py` решает, какие страницы отправить в Docling. Структурный ответ преобразуется напрямую из блоков Docling. При выборе кандидата заменяется страница целиком, с сохранением первичных наблюдений. Найденная детектором область сложной таблицы может остаться отложенной, пока её ячейки нельзя подтвердить.

## Быстрый запуск на Windows

Установите Python 3.12 и Tesseract с языковыми данными `eng` и `rus`. Проверьте, что `tesseract --list-langs` показывает оба языка. Если Tesseract не в `PATH`, задайте `BEAR_TESSERACT` с путём к исполняемому файлу.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
.\.venv\Scripts\bear-pdf-ocr.exe .\document.pdf -o .\document.json
.\.venv\Scripts\python.exe -m pytest -q
```

Выходной файл должен быть новым. Без `-o` результат выводится в stdout.

### Независимый OCR и детектор таблиц

```powershell
py -3.12 -m venv .ocr-venv
.\.ocr-venv\Scripts\python.exe -m pip install -r requirements-paddle-ocr.txt
.\.ocr-venv\Scripts\hf.exe download PaddlePaddle/PP-OCRv5_mobile_det --revision 0d63e78e2b680928f6b1747d76a08db6e645efb7 --include 'inference.*' --local-dir models/paddle/PP-OCRv5_mobile_det
.\.ocr-venv\Scripts\hf.exe download PaddlePaddle/eslav_PP-OCRv5_mobile_rec --revision 7553801264d3379d8d2e854971989e5e26c22e03 --include 'inference.*' --local-dir models/paddle/eslav_PP-OCRv5_mobile_rec
.\.ocr-venv\Scripts\hf.exe download PaddlePaddle/PP-DocLayout_plus-L --revision aa52b8528c84f9b1a34ac3a88fe0e576edb9d11d --include 'inference.*' --local-dir models/paddle/PP-DocLayout_plus-L
$env:BEAR_PADDLE_PYTHON = (Resolve-Path .\.ocr-venv\Scripts\python.exe).Path
$env:BEAR_PADDLE_MODELS = (Resolve-Path .\models\paddle).Path
$env:BEAR_PADDLE_LAYOUT_MODEL = (Resolve-Path .\models\paddle\PP-DocLayout_plus-L).Path
```

Детектор таблиц можно отключить, убрав `BEAR_PADDLE_LAYOUT_MODEL`. Модели загружаются при подготовке окружения. Обработчик читает локальные файлы весов.

### Постраничный резерв Docling

```powershell
py -3.12 -m venv .docling-venv
.\.docling-venv\Scripts\python.exe -m pip install -r requirements-docling.txt
.\.docling-venv\Scripts\docling-tools.exe models download -o .\models\docling layout tableformer
$env:BEAR_DOCLING_PYTHON = (Resolve-Path .\.docling-venv\Scripts\python.exe).Path
$env:BEAR_DOCLING_ARTIFACTS = (Resolve-Path .\models\docling).Path
```

Включённые компоненты используются при следующем вызове CLI. `BEAR_PDF_DOCLING=0` отключает резерв. В обеих конфигурациях Tesseract должен иметь доступ к языковым данным.

## Выходной формат

JSON содержит `schema_version`, `context`, `parser_profile`, `blocks`, `attachments` и `warnings`. `context` включает имя документа и SHA-256. Блоки упорядочены и имеют тип `TEXT`, `TABLE` или `IMAGE`.

| Поле блока | Содержимое |
|---|---|
| `id`, `order`, `type`, `role`, `text` | Идентификатор, положение в результате, тип и содержимое |
| `source.page`, `source.bbox` | Страница с единицы и прямоугольник `[x0, y0, x1, y1]` в пунктах PDF от верхнего левого угла, если известен |
| `metadata.ocr_words`, `metadata.ocr_verification` | Слова Tesseract, а при включённой сверке оба чтения, confidence, координаты и расхождения |
| `metadata.table_rows` или `metadata.table_cells` | Строки таблицы либо канонические ячейки, включая позицию и объединения |
| `metadata.visual_context`, `metadata.image_ocr` | Исходная подпись и ссылки на соседние блоки, наблюдения OCR по изображению |
| `metadata.pdf_route`, `metadata.unresolved` | Выбранный маршрут и признаки необходимости ручной проверки |

`pdf_route.status` показывает результат эвристических проверок страницы. Для решения о ручной сверке учитывайте также `unresolved`, предупреждения OCR и `source_fidelity=unverified`. Совпадение двух OCR на скане оставляет требование проверить исходные реквизиты.

## Лимиты и проверка

CLI ограничивает вход 50 МиБ и PDF до 500 страниц. Растр страницы ограничен 30 млн пикселей. PaddleOCR получает до 60 секунд на область, до восьми областей и 120 секунд от первой попытки на документ. Процесс контролируется по суммарной памяти дерева, с порогом 2 ГиБ. Детектор таблиц обрабатывает до двух страниц. Docling запускается для ограниченного числа проблемных страниц.

Экспорт содержит исходники PDF/OCR из BEAR_BGE_M3, CLI и тесты. Конфигурация, веса и оригиналы документов готовятся локально.

В чистом окружении Python 3.12 прошли 34 теста. На отдельном CPU-прогоне экспортированный код подтвердил одинаковое чтение контрольного скана Tesseract и PaddleOCR, восстановление 12 из 12 ячеек безрамочной таблицы, обнаружение области научной таблицы, OCR встроенной надписи с привязкой подписи и разбор размеченной таблицы через Docling. Рукописные реквизиты сложного приказа и сложные таблицы требуют сверки с оригиналом.
