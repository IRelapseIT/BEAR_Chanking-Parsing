# Воспроизведение сравнения на этом компьютере

Основной отчёт: `../../docs/MODEL_COMPARISON_2026-11-06.md`.
Здесь сохранены те же вопросы, фрагменты и результаты обеих моделей.
Скрипты сравнения используют одинаковый серверный адаптер из этого каталога,
а application-код импортируют из соответствующей папки проекта.
Копия адаптера — тестовая инфраструктура для одинаковых условий, не другой backend.

Исходная рабочая папка не изменяется. Все новые состояния и временные Qdrant
создаются рядом со скриптом. Для нового запуска безопаснее скопировать этот
каталог в пустую папку, чтобы не перезаписывать сохранённые результаты опыта.
Нативный Qdrant и модели ожидаются по указанным в скрипте локальным путям.
Полная установка на другом устройстве этим не подтверждается.

```powershell
cd 'D:/CODEX/BEAR/BEAR_BGE_M3/verification/model-comparison'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:HF_HUB_OFFLINE='1'
$env:PYTHONPATH='D:/CODEX/BEAR/upgrade/embedding-ml;D:/CODEX/BEAR/BEAR_BGE_M3/bear-text-beta/src;D:/CODEX/BEAR/BEAR_BGE_M3/bear-fusion-lab/src'
& 'D:/CODEX/BEAR/upgrade/.venv/Scripts/python.exe' run_model.py bge
& 'D:/CODEX/BEAR/upgrade/.venv/Scripts/python.exe' semantic_checks.py bge

$env:PYTHONPATH='D:/CODEX/BEAR/upgrade/embedding-ml;D:/CODEX/BEAR/BEAR_Chanking-Parsing/bear-text-beta/src;D:/CODEX/BEAR/BEAR_Chanking-Parsing/bear-fusion-lab/src'
& 'D:/CODEX/BEAR/upgrade/.venv/Scripts/python.exe' run_model.py qwen
& 'D:/CODEX/BEAR/upgrade/.venv/Scripts/python.exe' semantic_checks.py qwen
```

Скрипты действительно загружают модели; каждая занимает несколько ГиБ памяти.
Запускайте их последовательно. Пример с HTTP 500 создаётся намеренно для проверки
резерва. Серверный пакет Infinity в этих опытах не используется.

Полная регрессия BGE-копии требует `BEAR_TEST_TOKENIZER`, иначе часть проверок
реального токенизатора будет пропущена:

```powershell
$env:PYTHONPATH='D:/CODEX/BEAR/upgrade/embedding-ml;D:/CODEX/BEAR/BEAR_BGE_M3/bear-text-beta/src;D:/CODEX/BEAR/BEAR_BGE_M3/bear-fusion-lab/src'
$env:BEAR_TEST_TOKENIZER='D:/CODEX/BEAR/BEAR_BGE_M3/models/bge-m3'
& 'D:/CODEX/BEAR/upgrade/.venv/Scripts/python.exe' -m pytest ../../bear-text-beta/tests ../../bear-fusion-lab/tests -q -p no:cacheprovider
```

В публикацию включены исходники и JSON-отчёты; полные vectors.npz исключены. Для текущей проверки Infinity предусмотрена небольшая публичная эталонная фикстура verification/workflow/bge-reference.json. Исторические команды сравнения могут требовать настройки путей и самостоятельной подготовки весов.
