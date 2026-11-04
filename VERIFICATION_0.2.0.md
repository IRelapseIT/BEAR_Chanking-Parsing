# BEAR0.2.0: заключительные проверки

Дата: 4 ноября 2026. Ветка: `feat/bear-0.2.0-fixes`. Базовый commit: `c2447d4ff89cb0f5e33006d505e6d0243930c292`. Версия обоих проверенных пакетов: `0.2.0`.

| Проверка | Фактический результат |
|---|---|
| Полный bear-text-beta/tests | **137 passed**,0 failures/errors/skips |
| Полный bear-fusion-lab/tests | **67 passed**,0 failures/errors/skips |
| Итого штатных и новых регрессий в этих двух наборах | **204 passed**; повторные промежуточные запуски не суммируются |
| Тот же корпус15 документов |15/15 successful, hashes и TEXT source coverage проверены |
| Чанки корпуса |5848, максимум320 E5 tokens полного входа |
| Native smoke |Настоящие Qwen3-Embedding-0.6B CPU и Qdrant Windows1.20.0,25 чанков/24 вопроса |
| Native Recall@5 / MRR@5 |1.000 / 0.979167 |
| Отказ upsert нового поколения |Старая active generation сохранена и доступна поиску |
| Retry нового поколения |Успешно активирует новое, прежнее удаляется после activation |
| Другой tenant/space |0 hits |
| Oversized Qwen input |EmbeddingError до HTTP; network_requests=0 |
| pip check |No broken requirements found |
| Ruff core и изменённый Fusion src/test_fix_runtime |All checks passed |
| compileall / git diff --check |Прошли |

В core пройдены HTTP TestClient и реальные spawn-process lifecycle tests, ранее недоступные в старой среде. Запуск проведён вне Windows sandbox, где TestClient зависал. В чистом venv согласованы FastAPI/Starlette/httpx; автозагрузка сторонних pytest plugins отключена. Единственное предупреждение core — deprecation Starlette TestClient/httpx, не ошибка теста.

## Исправления, проверенные регрессиями

- Происхождение TEXT проверяется независимо от strategy; corrupt external/mixed spans и поддельный section_path отвергаются.
- Remote identity/document/profile/generation mismatch, oversized response, неверные UUID и невалидный профиль получают явный отказ; положительный remote contract проходит.
- Expired last lease становится failed; stale complete/fail/publication запрещены; heartbeat и SQLite migration проверены, включая два настоящих worker-процесса.
- Begin старой версии не отменяет принятую новую. Успешная и аварийная индексация проверены отдельно.
- Qdrant connect/read errors, create race и schema mismatch обрабатываются согласно контракту.
- Readiness не сообщает готовность модели при outage; offline parse/chunk и online index различаются.
- BOM upload→parse проходит; произвольный binary по-прежнему отклоняется. HTML br/inline fragments/literal code сохраняются.
- Ошибки demo write/submit освобождают очередь; malformed body получает4xx; completed результаты восстанавливаются после restart, interrupted задачи terminal.
- Контекст длинных headers сохраняет qualifier/year/unit/currency и delimiter, имеет полный source proof; значение не проходит с подменёнными spans/context.
- PDF two-column/mixed bands и OCRcrop pixel→PDF проверены независимыми fixtures.

## Данные и метод

Основной корпус неизменён относительно предыдущего сравнения:10 сохранённых публичных документов и5 controlled regressions. Все15 SHA256 сверены. Проверены конечные CELL_END/TABLE_END/SENTINEL_END, точная последовательность непробельных символов TXT и отсутствие пропусков/дублей TEXT spans. Прежние результаты comparison не перезаписаны. Успешный запуск не выдаётся за доказательство полной точности PDF/OCR.

Native smoke выполнен с реальными весами Qwen revision`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` через совместимый локальный endpoint и **IndexingService нового BEAR**, а не через сторонний store-service. Использован ограниченный input budget512, target96 и8 authored RU/EN документов/24 вопроса. Это малый smoke, не независимый holdout и не доказательство выигрыша одной модели/чанкера. При upsert failure ошибка инъецирована намеренно; существующая generation действительно оставалась на живом сервере.

## Повторение

Из корня checkout после установки по RELEASE_0.2.0.md:

```powershell
$env:BEAR_TEST_TOKENIZER='PATH_TO_PREPARED_E5'
python -m pip check
python -m pytest bear-text-beta/tests -q
python -m pytest bear-fusion-lab/tests -q
bear --model-dir "$env:BEAR_TEST_TOKENIZER" doctor --require-model
```

Полный локальный evidence находится рядом с checkout в `D:/CODEX/BEAR/upgrade`:logs/core-final.xml, fusion-final.xml, corpus-final-summary.json, live-results.json, oversize.json, final-summary.json. Scripts run_corpus.py/live_bear.py/verify_release.py фиксируют реальные пути этой машины; перед переносом нужно передать/изменить ресурсы. Публичный компактный итог без документов: [verification/0.2.0/results.json](verification/0.2.0/results.json).

CI workflow добавлен, но не запускался в GitHub. Linux matrix, чистая установка отдельного Dedoc/ML стека с нуля, native sparse/reranker, production нагрузка и universal PDF fidelity не заявляются проверенными. Прежний parser checkout используется только неизменённым и pinned. Docker не запускался; временные Qwen/Qdrant серверы остановлены после проверки.

Миграция состояния требует остановки старых workers; подробности и ограничения в [RELEASE_0.2.0.md](RELEASE_0.2.0.md).
