# BEAR Text Beta

TXT/HTML/DOCX → Dedoc 2.8 в отдельном контейнере; MD → `markdown-it-py`.
Адаптивный чанкинг использует Chonkie 1.7.0 в отдельном контейнере. Текст поиска хранится
в `payload.text` Qdrant. PDF/OCR/таблицы подключаются через порты. Подробная схема и
ограничения: [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md); сигнатуры модулей:
[MODULE_CONTRACTS_V3.md](MODULE_CONTRACTS_V3.md).

## Запуск

Все команды выполняются из папки `bear-text-beta`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m bear_text_beta.cli prepare-model
.\.venv\Scripts\python.exe -m bear_text_beta.cli --offline process examples/instruction.md --document instruction --output artifacts/instruction.json
```

Локальный запуск без Dedoc рассчитан на MD и тестовый режим. Для TXT/HTML/DOCX запустите
контейнер Dedoc и задайте `BEAR_DEDOC_URL`; для полного HTTP-стека используйте Compose ниже.

`prepare-model` скачивает только токенизатор и конфигурацию, фиксируя Hugging Face commit и контрольные суммы.
Повторный вызов сохраняет ту же ревизию. Для запуска Infinity нужны веса:

```powershell
.\.venv\Scripts\python.exe -m bear_text_beta.cli prepare-model --weights
.\.venv\Scripts\python.exe -m bear_text_beta.cli prepare-images
docker compose --env-file compose.lock.env --profile api up -d --build
.\.venv\Scripts\python.exe -m bear_text_beta.cli process examples/accounts.txt --document accounts --output artifacts/accounts.json
.\.venv\Scripts\python.exe -m bear_text_beta.cli index artifacts/accounts.json
.\.venv\Scripts\python.exe -m bear_text_beta.cli search "Как восстановить пароль?"
```

`prepare-images` скачивает публичные образы Infinity CPU и Qdrant и записывает их digest в `compose.lock.env`.
Compose требует этот файл: незакреплённые образы автоматически не подставляются.
Compose запускает API, процессинговый worker, Dedoc, Chonkie, Infinity и Qdrant.
Альтернатива для разработки — локальный API с работающими внешними сервисами:

```powershell
.\.venv\Scripts\python.exe -m uvicorn bear_text_beta.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Документация API: `http://127.0.0.1:8000/docs`. `/beta/process` принимает multipart-файл и JSON-строку `context`:

```json
{"tenant_id":"local","space_id":"demo","document_id":"accounts","document_version":1,"filename":"accounts.txt"}
```

`POST /beta/index`: `{"generation_id":"UUID из ответа process"}` либо `{"result":{...ProcessingResult...}}`.
`POST /beta/search`: `{"query":"Как восстановить пароль?","tenant_id":"local","space_id":"demo","top_k":5}`.
`GET /health`: готовность приложения, модели и внешних зависимостей.
Файл до 8 MiB обрабатывается синхронно. Для большего `POST /beta/process` возвращает HTTP 202
с `job_id`; состояние доступно по `GET /beta/jobs/{job_id}`. Оригинал сохраняется через
`ObjectStorage` при том же чтении, которым подсчитываются размер и SHA-256.

## Чанкинг

- `adaptive` (по умолчанию): маленький участок целиком; список или несколько блоков раздела — structural;
  длинная связная проза — Chonkie SemanticChunker; код и прочие случаи — RecursiveChunker.
  Один `section_path` без пригодных границ не включает structural.
- Сбой семантического инференса включает резервную ветвь; причина находится в `fallback_reason` и `warnings`.
- Предел участка семантического анализа — 2048 токенов. Более длинные неструктурированные участки идут
  в резервную ветвь с `semantic_budget_exceeded`; это явно ограниченная CPU-политика беты.
- Проверяются полный вход модели, источники, порядок и покрытие текста. Если библиотека изменила текст,
  SafeTextSplitter делит исходные Unicode-подстроки. Ничего не усекается молча.
- Цель — 320 токенов, максимум — 480, overlap — 0. Контекст заголовков ограничен 64 токенами;
  полные заголовки сохранены в метаданных.
- `--mode fixed|token|semantic` используется для сравнений и исходного ТЗ; HTTP API всегда применяет `adaptive`.
- `--offline` отключает предварительный инференс, сохраняя реальный токенизатор. Индексация в этом режиме
  недоступна. `--test-counter` — только демонстрация и тесты без скачивания токенизатора; её результат нельзя индексировать.

## Результаты, сравнение и качество

```powershell
.\.venv\Scripts\python.exe -m bear_text_beta.cli compare artifacts/a.json artifacts/b.json
.\.venv\Scripts\python.exe -m bear_text_beta.cli evaluate examples/evaluation.json --output artifacts/evaluation.json
```

`compare` сравнивает источники, границы, размеры, стратегии, покрытие и диагностику общей схемы v1.
`evaluate` реально индексирует один корпус тремя стратегиями и измеряет document Recall@5, MRR,
время, Python peak allocations и число входов/токенов инференса. Требует работающих Infinity и Qdrant;
не заменяет отсутствующий инференс случайными vectors. Python allocations не равны RAM всего процесса
или памяти контейнера модели. Демонстрационный корпус мал; используйте документы команды и размеченные вопросы.
Точность отдельных SourceSpan проверяется тестами; автоматическая retrieval-метрика беты оценивает документы.

## Хранение и повторы

`state/manifest.sqlite3` хранит checkpoints, задания/outbox и активные поколения. Тексты выдачи читаются
из Qdrant. SQLite checkpoint нужен для повторной индексации без изменения границ.

```powershell
.\.venv\Scripts\python.exe -m bear_text_beta.cli index --generation UUID
```

Поколение активируется только после полной записи. Неудача оставляет предыдущую версию доступной.
Три неудачных попытки индексации завершают задание; новый эксперимент создаёт новое поколение.
Та же `document_version` не может содержать другой checksum: при изменении файла увеличивайте `--version`.
`OutboxPublisher` и `IndexingWorker` реализуют тестируемые контракты повторов/ACK. Реальные PostgreSQL,
Redis Streams, ACL/OIDC и MinIO — интеграции команды; API беты рассчитан на локальный запуск.
Индексация сериализуется одним runtime; запускайте один API worker и не индексируйте параллельно из другого процесса.

## Форматы и ограничения

TXT/HTML перед Dedoc проходят строгое декодирование UTF-8/BOM; CP1251 включается флагом
`--allow-cp1251` либо `BEAR_ALLOW_CP1251=1`. MD обрабатывается `markdown-it-py`.
TXT/HTML/DOCX обрабатываются Dedoc. Dedoc-вложения, которые удалось извлечь, сохраняются через
локальный `ObjectStorage` beta; командный MinIO adapter ещё не подключён.
Таблицы и изображения без внешнего обработчика остаются `deferred_blocks`; результат помечается `partial`.
Надписи в изображениях распознаёт OCR-модуль команды. Лимиты: файл 50 MiB, распакованный DOCX
200 MiB, вложений 100 / 10 MiB каждое / 50 MiB суммарно, таблица inline до 1 MiB. Число страниц
DOCX проверяется по доступным метаданным, но без рендеринга не гарантируется.

PDF adapter регистрируется через `ParsingPipeline(extra_parsers={".pdf": adapter})`.
Table/OCR adapter подключается в `ChunkingService(external_processors={"TABLE": adapter, "IMAGE": adapter})`.
Точная модель данных и ответственность модулей описаны в `ARCHITECTURE.md`.

## Проверка и окружение

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=state/pytest-local
.\.venv\Scripts\python.exe -m ruff check src tests
```

Тесты используют реальные Chonkie/парсеры и контролируемые embeddings/HTTP responses. Один дополнительный
тест использует подготовленный настоящий E5 tokenizer; без кеша он пропускается. Тесты сами модели не скачивают.
В ограниченной Windows-песочнице FastAPI TestClient требует разрешённый локальный socketpair.

Параметры окружения: `BEAR_MODEL_DIR`, `BEAR_STATE_DIR`, `BEAR_INFINITY_URL` (по умолчанию
`http://127.0.0.1:7997/v1`), `BEAR_INFINITY_MODEL` (`bear-e5-small`), `BEAR_QDRANT_URL`,
`BEAR_QDRANT_COLLECTION`, `BEAR_DEDOC_URL`, `BEAR_CHUNKER_URL`, `BEAR_INFINITY_API_KEY`,
`BEAR_QDRANT_API_KEY`, `BEAR_OFFLINE`, `BEAR_TEST_COUNTER`.
API ключи не записываются в результаты или lock-файл. Для нового embedding-профиля используйте новую
коллекцию и переиндексацию; vectors разных профилей исключаются фильтром.

Зависимости закреплены в `requirements.lock`; runtime-манифест записывает версии и профили.
Для контролируемого обновления сначала меняйте окружение, запускайте проверки, затем `bear lock`.
