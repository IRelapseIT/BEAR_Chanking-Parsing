# BEAR Fusion Lab — первая экспериментальная сборка

Текущее обновление: [v2, решения и проверки](REVISION_V2.md).

Обновление 03.11.2026: [локальное демо и запуск](DEMO.md).
Добавлены PDF/CPU OCR, поисковые табличные фрагменты, эталон и ограниченная очередь.
Ниже сохранено описание первоначальной сборки; актуальный запуск и ограничения — в DEMO.md.

Отдельный проект; исходники `bear-text-beta` и участника не изменены.
Задача этапа: реальный парсер участника → общий контракт → наш автоматический чанкинг.
Это НЕ завершённая универсальная RAG-система и не результат сравнения качества поиска.

## Зафиксированная исходная версия

Проверка GitHub 02.11.2026 обнаружила новую ветку `test_dedoc_qdrant`:
`a48d88d21ba6ece0fc37eb1f91f18905705dec7f` (коммит от 01.11).
Обновлённая 8.0: `bfe2cc08dad9d98151d74d6f06bdf0edf8e3afd8`.
В новой ветке исправлен контракт raw/embedding text, переписано подключение semantic embeddings,
добавлены Qdrant, sparse и reranking. Старые замечания об отсутствии этих компонентов неприменимы.
В сравнении двух этих веток файлы парсеров не менялись.

Рабочая копия: `../research/RAG_VERSION_1-20261102`.
Worker проверяет commit и отсутствие изменений отслеживаемых файлов перед запуском.
Права на исходный проект не предполагаются: перед публикацией производного дистрибутива
нужно согласовать с автором лицензию/атрибуцию. Сейчас код подключается из его отдельной копии.

## Реализованный путь

`FusionPipeline.process_path(path: Path, context: DocumentContext, profile: ProcessingProfile) -> dict`

1. `FileValidator.validate_and_store` — ограниченная загрузка, SHA256, существующий ObjectStorage.
2. Материализация временного файла; bounded ZIP-проверка DOCX.
3. `reject_known_table_loss(source: Path) -> None` — отказ на известных размерах обрезания HTML/DOCX-таблиц.
4. TXT/HTML/DOCX: `ParserProcess.parse(source: Path) -> dict` вызывает неизменённый `DedocParser`
   в отдельном Python 3.11. Markdown: прямой `MDParser` после эксперимента E02.
5. `map_tree(payload: dict, context: DocumentContext, limits: FileLimits) -> ParsedDocument`:
   полный обход дерева, сохранение заголовков и текста корня, таблицы отдельными блоками,
   контекст предков передаётся локально, а не общим изменяемым стеком.
6. Существующий `ChunkingService.process` из беты: whole / structural / semantic / recursive / safe repair.
7. JSON-конверт с исходным объектом, parser runtime, hash кода беты и результатом чанкинга.

Источники для upstream — адрес узла дерева, **не выдуманные страницы/координаты оригинала**.
Таблицы сохраняются как deferred; их содержимое не считается проиндексированным.
`accepted_for_indexing=false` всегда: индексация намеренно не подключена на этом этапе.
`result.completeness` описывает обработку блоков, а не доказанную полноту извлечения исходного файла.

## Запуск в текущем workspace

Работа из папки `bear-fusion-lab`, PowerShell:

```powershell
$env:HF_HUB_OFFLINE='1'
& '../bear-text-beta/.venv/Scripts/python.exe' run.py artifacts/fixtures/sections.docx `
  --upstream '../research/RAG_VERSION_1-20261102' `
  --parser-python '../research/rag-dedoc3-env/Scripts/python.exe' `
  --tokenizer '../bear-text-beta/models/e5' --state state `
  --tenant demo --space lab --document docx-demo --offline --output artifacts/docx-demo.json
```

Результат не перезаписывается: для повторного запуска выбери новый output.
`--offline` явно включает резерв при необходимости семантического инференса.
Вместо него можно указать `--infinity-url http://localhost:7997/v1`; этот режим здесь ещё не проверен.
Никакие фиктивные embeddings за реальные не выдаются.

Для установки пакета нужны editable-зависимости из workspace:
`pip install -e ../bear-text-beta -e .` в отдельном Python 3.12 окружении.
Это не инструкция установки Dedoc/Docling: parser worker требует своего окружения.
Текущие успешные прогоны использовали уже существующее исследовательское Python 3.11 окружение;
полная установка новых requirements участника и Docling не проверены.

## Воспроизведение экспериментов

```powershell
& '../bear-text-beta/.venv/Scripts/python.exe' experiments/compare.py
& '../bear-text-beta/.venv/Scripts/python.exe' experiments/native_markdown.py
& '../bear-text-beta/.venv/Scripts/python.exe' -m pytest tests -q -p no:cacheprovider
```

`compare.py` обрабатывает реальные синтетические файлы upstream-парсером и настоящим
локальным токенизатором E5; embeddings недоступны намеренно для проверки fallback.
Текст, структура, контрольные маркеры, фактические размеры чанков и runtime сохраняются в artifacts.
Это повторяемый функциональный корпус, не статистический benchmark и не оценка retrieval.

## Ограничения

- Только TXT/MD/HTML/DOCX в маршруте сборки; PDF/OCR/Excel пока не включены.
- Отдельный процесс не является sandbox. Ограничение 120 секунд и размера ответа не заменяет
  cgroups/job objects, контроль памяти, запрет сети и остановку всех дочерних конвертеров.
- Верхний лимит JSON проверяется после завершения worker; потокового чтения дерева пока нет.
- Preflight закрывает известные пороги обрезания таблиц, но не гарантирует отсутствие других потерь.
- Полнота исходника unverified; upstream не извлекает вложения в нашем контракте.
- Бета подключена как существующее ядро, не скопирована. Hash её исходников пишется в CLI-результат.
- Нет измерений peak RSS, поисковой точности, CPU-семантического инференса или полного Qdrant E2E.

Следующие решения — только после новых экспериментов: исправление табличного renderer с
сохранением escaped pipes/merged cells, прямой адаптер Docling, bounded parent retrieval,
тесты отказов процесса/памяти, затем одинаковая embedding-модель и размеченные retrieval-запросы.
