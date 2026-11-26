# Локальный запуск BEAR на Windows

Стенд запускает четыре процесса на localhost. FastAPI обслуживает веб-интерфейс и API, worker обрабатывает документы, Infinity считает dense-векторы BGE-M3 на CPU, Qdrant хранит индекс. OCR и Docling запускаются по мере необходимости отдельными процессами.

## Установка с нуля

Понадобятся Python 3.12, PowerShell и доступ к PyPI, Hugging Face, репозиторию релизов Qdrant и CPU-колёсам PyTorch на время установки. Для PDF-сканов установите Tesseract и пакеты языков `eng`, `rus`, `osd`. Для DOC, ODT и RTF установите LibreOffice.

В корне проекта выполните из PowerShell:

```powershell
./setup-workflow.ps1 -Python 'C:/path/to/python.exe' -DownloadQdrant -PrepareModels -WithPaddle -WithDocling -Start
```

Если `python` в PATH уже указывает на Python 3.12, параметр `-Python` можно опустить. Если Tesseract и LibreOffice установлены вне PATH, передайте их исполняемые файлы через `-Tesseract` и `-LibreOffice`. Параметры `-WithPaddle` и `-WithDocling` добавляют отдельные окружения и модели для резервных маршрутов PDF. Для текстовых документов достаточно API, Infinity и Qdrant. Параметр `-Start` запускает сервисы только после проверки окружений и артефактов.

Скрипт проверяет версию каждого существующего venv и выполняет `pip check` после установки. Веса скачиваются отдельно от обработки документов. Скачивание моделей использует закреплённые ревизии из [prepare_stand_models.py](../scripts/prepare_stand_models.py). Для BGE-M3 проверяется SHA-256 весов и фиксируются хеши файлов токенизатора. Для моделей Paddle проверяются три файла каждого снимка. Docling использует закреплённые снимки Heron и TableFormer. Официальный архив Qdrant 1.20.0 проверяется по SHA-256 перед распаковкой.

## Повторный запуск

Если установщик создал `runtime/stand-local.json`, используйте его во всех командах управления. Для конфигурации по умолчанию параметр `--config` можно опустить.

```powershell
./.venv/Scripts/python.exe scripts/services.py doctor --config runtime/stand-local.json
./.venv/Scripts/python.exe scripts/services.py start --config runtime/stand-local.json
./.venv/Scripts/python.exe scripts/services.py status --config runtime/stand-local.json
./.venv/Scripts/python.exe scripts/services.py stop --config runtime/stand-local.json
```

Откройте http://127.0.0.1:8002/ или http://127.0.0.1:8002/docs. `doctor` показывает проверенные версии интерпретаторов, ревизию BGE-M3 и наличие инструментов и артефактов. `GET /live` проверяет ответ API. `GET /ready?capability=index` проверяет готовность конвейера индексации. Готовность артефактов OCR подтверждает конфигурацию, а результат распознавания проверяется на контрольных сканах.

Для запуска в текущем терминале выполните `scripts/services.py run` с тем же конфигом. `Ctrl+C` или `stop` с тем же конфигом остановят зарегистрированные процессы. Перед остановкой сверяются PID, время создания и командная строка. Если порт занят, запуск сообщит об этом и сохранит существующий процесс. Данные сохраняются после остановки.

## Уже установленные инструменты и изоляция экземпляров

`--config` или `BEAR_LOCAL_CONFIG` задаёт JSON с локальными переопределениями [config.local.json](../config.local.json). Относительные пути считаются от корня проекта. Существующий локальный файл настройки установщик сохраняет. Для второго стенда задайте отличающиеся порты и свои каталоги состояния.

| Ключ | Назначение |
| --- | --- |
| `ports.api`, `ports.embedding`, `ports.qdrant`, `ports.qdrant_grpc` | Локальные порты, по умолчанию 8002, 7998, 6335, 6336. |
| `runtime_dir`, `state_dir` | PID-реестр, журналы и данные Qdrant в первом каталоге. SQLite, оригиналы и результаты обработки во втором. |
| `api_python`, `infinity_python`, `model_dir`, `qdrant_exe` | Переиспользование установленных окружений, BGE-M3 и исполняемого файла Qdrant. |
| `tesseract_exe`, `libreoffice_exe` | Явные пути к инструментам вне PATH. |
| `docling_python`, `docling_artifacts` | Python 3.12 с Docling и каталог с уже загруженными снимками. |
| `paddle_python`, `paddle_models`, `paddle_layout_model` | Python 3.12 с PaddleOCR, каталог двух OCR-моделей и каталог необязательной модели разметки. |
| `collection` | Имя коллекции Qdrant, привязанной к выбранному профилю. |
| `package_dirs` | Необязательный список дополнительных локальных каталогов Python-пакетов. |

## Контрольная загрузка PDF

При работающем стенде из корня проекта:

```powershell
./.venv/Scripts/python.exe verification/stand_pdf_smoke.py --output runtime/verification/pdf.json
```

Для двух дополнительных публичных примеров передайте `--public-corpus C:/path/to/corpus` с файлами `ocr-linn.pdf` и `attention.pdf`. Скрипт использует отдельный тестовый Space. В отчёте есть происхождение и SHA-256 документов, наблюдаемые фазы, сохранность оригинала, доступность страниц, покрытие извлечённого текста, ранг ответа на контрольный вопрос и измеренные времена. Пагинация оригинала, покрытие извлечённых блоков и точность OCR отражают разные свойства.

