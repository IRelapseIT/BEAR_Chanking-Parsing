# BEAR Fusion 0.2.0

Текущая сборка использует общий BEAR core для модели документа, токенизации, чанкинга и проверки источников. Полная инструкция установки находится в [корневом README](../README.md).

В этой версии исправлены освобождение очереди после ошибок записи/submit, восстановление состояния demo после перезапуска, проверка JSON запросов, контекст длинных заголовков таблиц, порядок колонок PDF и преобразование координат OCR. История и миграция: [RELEASE_0.2.0.md](../RELEASE_0.2.0.md).

Поддерживаемые demo входы: TXT, MD, HTML, DOCX и PDF. TXT/MD/HTML используют локальные парсеры, DOCX — отдельный pinned parser process. PDF использует pdfplumber и Tesseract при необходимости. Layout определяется эвристикой; для inferred/неоднозначных областей сохраняются предупреждения и структурированное качество. Docling не является текущим production backend этой версии.

`accepted_for_indexing=false` сохраняется: успешное покрытие извлечённых блоков не доказывает точность всего источника. Demo использует локальный лексический поиск; сквозной vector indexing проверяется отдельно через core.

Исходный demo запускается из checkout:

```powershell
python bear-fusion-lab/demo.py --upstream 'ПУТЬ_К_RAG_VERSION_1' --parser-python 'ПУТЬ_К_PYTHON_ПАРСЕРА' --tokenizer 'ПУТЬ_К_E5'
```

Установите `bear-fusion-lab[demo]` для PDF/runtime зависимостей, `[acceptance]` для PDF fixtures. Tesseract — отдельный исполняемый файл, задаваемый `BEAR_TESSERACT`. Модель/сервис inference задаются явно; без него работает обозначенный fallback. Package включает библиотеку, demo UI/launcher являются source-only файлами.

Проверки:

```powershell
$env:BEAR_TEST_TOKENIZER='ПУТЬ_К_ПОДГОТОВЛЕННОМУ_E5'
python -m pytest bear-fusion-lab/tests -q
```

Без tokenizer часть ресурсных тестов явно пропускается. Для полной приёмки используйте подготовленный tokenizer, а не считайте такой skip успешным модельным тестом. Старые исследования — [FINAL_INTEGRATION_REPORT.md](FINAL_INTEGRATION_REPORT.md); их цифры не описывают автоматически новую версию.
