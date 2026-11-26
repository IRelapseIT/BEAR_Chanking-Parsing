# BEAR — локальный стенд документов

BEAR принимает документы через FastAPI, сохраняет оригинал, извлекает канонический ParsedDocument, создаёт Chunk с источниками, кодирует embedding_text через Infinity BGE-M3 и записывает текст и векторы в Qdrant. Частичный результат с пригодным текстом автоматически доступен для индексации. Предупреждения сохраняются отдельно от статуса выполнения и готовности поиска.

## Библиотеки и компоненты

| Библиотека или компонент | Назначение |
| --- | --- |
| Python 3.12, FastAPI, Pydantic, Uvicorn | API, контракты и локальный интерфейс |
| SQLite, ObjectStorage | Очередь, оригиналы, checkpoints, leases и поколения |
| markdown-it-py | Markdown, код, списки и таблицы |
| BeautifulSoup, lxml | HTML и защищённый XML |
| ZIP/XML | DOCX и проверка контейнеров |
| csv, openpyxl, xlrd | CSV/TSV, XLSX и XLS |
| LibreOffice | Ограниченная конвертация DOC, ODT, RTF в DOCX |
| pdfplumber, PDFium, Pillow | PDF, геометрия, растр и изображения |
| Tesseract | Основное OCR |
| PaddleOCR, Docling | Дополнительное OCR и резервный разбор страниц |
| Chonkie 1.7.0, Razdel | Семантическое и рекурсивное разбиение |
| transformers | Реальный токенизатор и контроль размера входа |
| Infinity, BAAI/bge-m3 | Dense embeddings, 1024 измерения |
| Qdrant | Dense и sparse поиск, фильтры и payload.text |

## Запуск и проверка

Подробная установка приведена в [LOCAL_SETUP.md](docs/LOCAL_SETUP.md).

Из PowerShell в корне проекта:

~~~powershell
./setup-workflow.ps1 -Python 'C:/path/to/python.exe' -DownloadQdrant -PrepareModels -WithPaddle -WithDocling -Start
~~~

Повторный запуск подготовленного окружения:

~~~powershell
./.venv/Scripts/python.exe scripts/services.py start --config runtime/stand-local.json
~~~

Интерфейс доступен по адресу http://127.0.0.1:8002/, OpenAPI по адресу http://127.0.0.1:8002/docs. Параметры путей задаются локальным конфигом. Остановка выполняется той же командой с действием stop.

## Форматы и поток данных

Реестр содержит TXT, MD, HTML/HTM, DOCX, PDF, CSV, TSV, XLSX, XLS, PNG/JPG/JPEG, XML, DOC, ODT и RTF. Для последних трёх нужен LibreOffice. [Матрица форматов](docs/STAND_FORMATS.md) описывает ограничения.

UploadFile и DocumentContext проходят ограниченную загрузку с SHA-256 в ObjectStorage. Парсер формирует ParsedDocument с упорядоченными ContentBlock, SourceLocator, attachments и warnings. Текстовые блоки маршрутизируются по цепочке whole → structural → semantic → recursive → safe repair. Наличие section_path само по себе не включает structural. Таблицы используют канонические ячейки и контролируемую упаковку строк. При техническом отказе специального обработчика доступный текст получает external_fallback с неопределёнными связями.

Chunk содержит text, embedding_text, token_count, strategy, fallback_reason, source_spans и section_path. Проверяется полный вход токенизатора, включая контекст. Сохранённый ProcessingResult доступен при отказе Infinity или Qdrant. Повтор индексации использует checkpoint. Перечанковка использует сохранённый разбор.

Поиск поддерживает dense, лексический sparse и RRF fusion. Опциональный HTTP reranker подключается через BEAR_RERANK_URL и BEAR_RERANK_MODEL. В базовом окружении модель реранкинга не загружена. Trace показывает реальные ветки, ранги, родителей, ограничения и время. Все ветки ограничены tenant/space и активными поколениями.

## Статусы и API

Поле outcome показывает completed, completed_with_warnings или failed. execution_state отражает очередь/worker, processing_phase — текущую фазу, completeness — полноту извлечённого результата, index.status и search_available — готовность поиска. Исторический status сохранён для совместимости.

Основные маршруты:

- POST /v1/documents — загрузка и автоматическая обработка.
- GET /v1/jobs и /v1/jobs/{id} — история и состояние.
- GET /v1/jobs/{id}/result — сохранённый результат.
- GET /v1/jobs/{id}/artifacts/{kind} — оригинал, JSON, Markdown, plain text, чанки, предупреждения, конфигурация и отчёт.
- GET /v1/jobs/{id}/page/{page} — растр страницы PDF/изображения.
- POST /v1/jobs/{id}/rechunk — новая конфигурация из сохранённого разбора.
- POST /v1/jobs/{id}/retry — повтор индексации.
- POST /v1/jobs/{id}/reprocess — повтор из оригинала.
- POST /v1/jobs/{id}/reassess — явная повторная приёмка старого needs_review.
- DELETE /v1/jobs/{id}/index — удаление только индекса.
- DELETE /v1/jobs/{id} — удаление выбранного прогона.
- POST /v1/search/trace — поиск с диагностикой и контрольным фрагментом.
- GET /v1/search/history — сохранённая история запросов.

Новые маршруты просмотра принимают tenant_id и space_id. Старые маршруты заданий допускают прежний вызов по UUID для совместимости. Стенд предназначен для доверенного localhost. Авторизация пользователей требует отдельного внедрения.

## Проверка вручную

Загрузите документ в разделе «Документы». После обработки откройте оригинал, блоки и предупреждения. В «Чанкинг» сравните text и embedding_text и выполните перечанковку. В «Индекс» проверьте активное поколение. В «Поиск» задайте вопрос и раскройте диагностику. В «Прогоны» сравните конфигурации. Просмотр блоков использует пагинацию, полный текст доступен в экспорте.

Полнота покрытия извлечённого текста и наличие страниц не доказывают точность OCR. XLS хранит доступные значения, исходные формулы BIFF могут быть недоступны. Конвертация офисных файлов может менять представление. Неинтерпретированные изображения остаются видимыми с предупреждениями.

Функции референса RAG_VERSION_1 воспроизведены собственной реализацией. Корневая лицензия референса отсутствует, его приложение в BEAR не копировалось.
