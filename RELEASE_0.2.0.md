# BEAR 0.2.0 — исправления и обновление

Дата: 4 ноября 2026. Ветка: `feat/bear-0.2.0-fixes`, исходная версия `main@c2447d4`.

## Исправления

- TEXT source validation больше не обходится strategy=external; смешанные TEXT/TABLE spans и посторонний контекст отклоняются.
- TABLE/IMAGE требуют зарегистрированного независимого валидатора, в том числе на входе индексации. Fusion TableProcessor предоставляет такой контракт.
- Remote chunker сверяет исходный документ/context/generation/profile, ограничивает ответ до JSON materialization и проверяет его локально. Без counter/profile обработка запрещена.
- SQLite processing leases получили owner/token/heartbeat; завершение, ошибка и публикация проверяют владение. Последняя expired попытка становится failed. Desired version не откатывается переупорядоченным start.
- Runtime использует source-preserving native TXT/HTML и общий table processor, когда установлен Fusion. Явное подключение Dedoc для текстовых форматов остаётся возможным.
- HTML br сохраняется как newline без склейки слов или добавления лишних пробелов между inline тегами. Upload принимает поддерживаемый decoder BOM UTF16/32 и продолжает отклонять binary.
- Qdrant ensure_collection корректно отображает transport failures и проверяет схему после гонки создания коллекции.
- Добавлены /live и /ready?capability=parse|chunk|index|search; /health больше не утверждает model_ready при недоступной модели.
- Demo проверяет JSON типы/имена файлов, освобождает очередь при write/submit failure, атомарно сохраняет state и восстанавливает завершённые результаты после restart; interrupted задания становятся failed.
- Длинный header таблицы сохраняет terminal qualifier, известные годы/валюту/единицы и разделитель; полный контекст и сокращение записаны в metadata. Канонические значения и header proof перепроверяются независимо от soft/hard budget.
- PDF порядок определяется по устойчивым gutters и full-width bands; inference/ambiguity видны в warnings. OCR сохраняет crop/scale/transform и page bbox.
- Shipped dedoc_mapper устраняет импорт отсутствующего dedoc_service из теста. `bear doctor` проверяет локальные ресурсы без запуска внешних приложений; обе версии пакетов —0.2.0.

## Установка

Python3.12. Из корня checkout:

```powershell
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -c constraints-py312-tested.txt -e './bear-text-beta[dev]' -e './bear-fusion-lab[demo,acceptance]'
./.venv/Scripts/python.exe -m pip check
./.venv/Scripts/bear.exe doctor
```

`constraints-py312-tested.txt` — точные версии проверенного Windows/Python3.12 окружения, полученные после успешного resolver/install. Это snapshot constraints, без полного hash-lock и без заявления о выполненной Linux CI проверке. Workflow `.github/workflows/tests.yml` добавляет Windows/Linux и отдельный tokenizer integration job; запуск workflow в GitHub ещё не выполнялся.

Подготовьте tokenizer через `bear --model-dir PATH prepare-model`, затем задайте `BEAR_TEST_TOKENIZER`. Runtime не скачивает модель молча. DOCX/Fusion требует отдельной неизменённой копии RAG_VERSION_1 commit `a48d88d21ba6ece0fc37eb1f91f18905705dec7f` и parser Python; чистая установка этого отдельного parser/ML стека не заменяется установкой core. Не смешивайте его старые NumPy/SciPy зависимости с ML inference environment.

## Обновление существующего состояния

1. Остановите старые workers и сохраните резервную копию SQLite state и оригиналов перед обновлением. Не запускайте unfenced старые workers вместе с0.2.0.
2. Новый SQLiteManifest под блокировкой добавляет `desired_version`, `claim_token`, `owner_id`, `lease_until`, `completed_at` и индекс claims. Старые записи и оригиналы сохраняются.
3. Старые running без token возвращаются в pending либо failed, если уже было3 попытки; причина `legacy_lease_recovered`. Ничего не удаляется автоматически.
4. Внешние реализации workers должны передавать `claim_token` в complete/fail и `processing_claim=(job_id,token)` в Services.process_stored. Встроенный process_worker уже делает это и heartbeat.
5. RemoteChunkingService теперь требует локальные counter/embedding_profile. Обновляйте client и chunker service согласованно.
6. Старые table checkpoint без нового full header-context proof нужно переработать из сохранённых оригиналов. Различие версии пакета входит в processing profile: несовместимый checkpoint не индексируется тихо как новый. Ранее активированные данные не удаляются этой миграцией; для смены модельного профиля используйте новую generation/collection.

## Заключительные проверки

Полный core и Fusion suites выполнены локально в новом venv с настоящим E5 tokenizer. Отдельно повторены15 документов исходного корпуса, проверены source hashes, TEXT span coverage, конечные маркеры длинных ячеек/таблиц и budget. Сквозная проверка core выполнена с настоящим Qwen3-Embedding-0.6B и живым Qdrant Windows, включая отказ upsert до activation и успешный retry. Итоговые числа, версии, границы и пути evidence находятся в `VERIFICATION_0.2.0.md`.

## Ограничения

PDF layout остаётся эвристикой, не универсальным распознаванием Docling. Полная точность OCR/таблиц/attachments требует независимой source verification; readiness Fusion остаётся false. Native sparse/reranker, полный распределённый broker, authentication/gateway и production нагрузка не объявляются проверенными. Process quotas library/CLI не заменяют OS isolation. Эти расширения остаются следующими этапами плана, а не скрытыми исправлениями этой версии.
