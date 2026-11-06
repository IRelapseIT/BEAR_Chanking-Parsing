"""Frozen BEIR SciFact subset with hard negatives, plus explicit authored controls."""
import csv
import hashlib
import io
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import httpx

ROOT = Path(__file__).resolve().parent
URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'


def main():
    archive = ROOT / 'scifact.zip'
    if not archive.exists():
        with httpx.Client(follow_redirects=True,timeout=60) as client, client.stream('GET',URL) as response:
            response.raise_for_status()
            data=bytearray()
            for block in response.iter_bytes():
                data.extend(block)
                if len(data)>32*1024**2:
                    raise ValueError('Corpus download exceeds 32MiB')
        archive.write_bytes(data)
    with ZipFile(archive) as z:
        documents={d['_id']:d for d in map(json.loads,z.read('scifact/corpus.jsonl').decode().splitlines())}
        queries={d['_id']:d for d in map(json.loads,z.read('scifact/queries.jsonl').decode().splitlines())}
        labels={}
        for row in csv.DictReader(io.StringIO(z.read('scifact/qrels/test.tsv').decode()),delimiter='\t'):
            if int(row['score'])>0:
                labels.setdefault(row['query-id'],set()).add(row['corpus-id'])
    rng=random.Random(20261106)
    chosen=sorted(rng.sample(sorted(labels),40))
    positives=set().union(*(labels[q] for q in chosen))
    token=lambda s:re.findall(r'\w+',s.lower())
    texts={k:d['title']+'\n\n'+d['text'] for k,d in documents.items()}
    counts={k:Counter(token(t)) for k,t in texts.items()}
    frequencies=Counter(t for bag in counts.values() for t in bag)
    average=sum(sum(b.values()) for b in counts.values())/len(counts)
    selected=set(positives)
    for q in chosen:
        terms=token(queries[q]['text'])
        def score(k):
            bag=counts[k]; length=sum(bag.values())
            return sum(math.log(1+(len(counts)-frequencies[t]+.5)/(frequencies[t]+.5))*bag[t]*2.2/
                       (bag[t]+1.2*(.25+.75*length/average)) for t in terms)
        candidates=sorted((k for k in texts if k not in labels[q]),key=lambda k:(-score(k),k))[:2]
        selected.update(candidates)
    selected.update(rng.sample(sorted(set(documents)-selected),24))
    output={'scope':'Reduced SciFact candidate pool, NOT full BEIR results; manually authored fictional controls are separate',
            'source':{'url':URL,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'seed':20261106,
                      'sampling':'40 test queries; all positives; top 2 BM25 nonrelevant abstracts per query; 24 random distractors'},
            'documents':[{'id':'scifact/'+k,'text':texts[k],'group':'scifact'} for k in sorted(selected)],
            'queries':[{'id':'scifact/'+q,'text':queries[q]['text'],'relevant_ids':['scifact/'+d for d in sorted(labels[q])],
                        'group':'scifact'} for q in chosen]}
    # Each item is fictional, independently labelled; paired policies make numeric/unit mistakes visible.
    controls=[
      ('refund','Магазин Норд возвращает деньги за обычный товар в течение 12 рабочих дней. Для бракованного товара срок составляет 7 календарных дней.',
       'Каков срок возврата денег в Норде за товар без брака?','ru'),
      ('refund-bad','Магазин Норд меняет бракованный товар за 7 рабочих дней. Это срок замены, а не возврата денежных средств.',
       'За сколько рабочих дней Норд заменяет товар с браком?','ru'),
      ('oncall','Команда Вега устраняет критические инциденты за 30 минут круглосуточно. Запросы низкого приоритета выполняются в течение двух рабочих дней.',
       'Как быстро Вега должна устранить критический инцидент ночью?','ru'),
      ('oncall-reply','Команда Орион отвечает на критический инцидент за 15 минут. Срок устранения проблемы составляет четыре часа.',
       'За какое время Орион должен устранить, а не просто подтвердить, критический инцидент?','ru'),
      ('vacation','Заявку на обычный отпуск подают минимум за 14 календарных дней. Для учебного отпуска достаточно пяти рабочих дней.',
       'За сколько дней сотрудник должен подать заявку на учебный отпуск?','ru'),
      ('backup','Резервные копии хранятся 30 суток в основном хранилище. Архивные копии хранятся 365 суток, восстановление занимает до 12 часов.',
       'Сколько времени допускается на восстановление из архивной копии?','ru'),
      ('backup-recovery','Основное хранилище восстанавливают за 15 минут. Этот срок не относится к архивным копиям.',
       'Как быстро нужно восстановить основное хранилище?','ru'),
      ('sales-table','Продажи, млн рублей. Столбцы: год; выручка; затраты. 2024; 120; 90. 2025; 150; 100.',
       'Какая выручка была в 2025 году, если затраты составили 100 миллионов?','ru'),
      ('sales-other','Продажи филиала: выручка 2024 года — 150 млн рублей; выручка 2025 года — 120 млн рублей. Затраты в 2025 году — 90 млн рублей.',
       'Какова выручка филиала в 2025 году при затратах 90 миллионов?','ru'),
      ('limits','Для проекта Север загрузка одного файла ограничена 20 МБ. На пользователя допускается 200 документов. Срок хранения — 90 дней.',
       'Какой максимальный размер одного загружаемого файла в проекте Север?','ru'),
      ('limits-team','В проекте Юг одному пользователю разрешено загрузить 20 документов. Каждый файл может занимать до 200 МБ.',
       'Сколько документов может загрузить один пользователь в проекте Юг?','ru'),
      ('go-cancel','В Go отмена context.Context сигнализирует горутинам через канал Done. Горутина должна сама обработать сигнал; автоматической остановки функции не происходит.',
       'Останавливает ли отмена контекста Go запущенную горутину автоматически?','ru'),
      ('go-channel','Закрытие канала Go не отменяет context.Context. После чтения всех буферизованных значений чтение возвращает нулевое значение и ok=false.',
       'Что вернёт чтение из закрытого канала после опустошения буфера?','ru'),
      ('go-sort','func sortedCopy(a []int) []int { b := append([]int(nil), a...); sort.Ints(b); return b }. Функция сортирует копию и сохраняет исходный срез.',
       'Какая функция возвращает отсортированный массив, сохраняя исходный срез Go?','ru'),
      ('go-inplace','func sortInPlace(a []int) { sort.Ints(a) }. Функция изменяет исходный срез; копия не создаётся.',
       'Какой вариант сортировки меняет переданный срез без создания копии?','ru'),
      ('tls','Для домена api.example.test TLS-сертификат хранится в /etc/bear/tls/api.pem. Приватный ключ находится в /etc/bear/tls/api.key.',
       'Где лежит приватный ключ домена api.example.test?','ru'),
      ('tls-other','Для домена admin.example.test сертификат хранится в /etc/bear/tls/admin.pem. Публичный сертификат api.example.test лежит в /etc/bear/tls/api.pem.',
       'Какой файл содержит публичный сертификат admin.example.test?','ru'),
      ('audit','Audit events are retained for 180 days. Debug logs expire after 7 days. Access to audit events requires the security-reviewer role.',
       'Какая роль требуется для доступа к журналу аудита?','cross_ru_en'),
      ('audit-short','Debug logs are retained for seven days and can be read by developers. These permissions do not grant access to audit events.',
       'How long are debug logs retained for developers?','en'),
      ('replicas','The Atlas database uses three synchronous replicas. Two acknowledgements are required before confirming a write; read-only replicas do not participate.',
       'How many acknowledgements are required to confirm an Atlas database write?','en'),
      ('retries','The client retries temporary HTTP failures twice after the initial request. Total attempts are limited to three. Authentication failures are not retried.',
       'What is the total number of client attempts, including the initial request?','en'),
      ('retries-other','The background worker allows two total job attempts, including the first execution. Authentication errors terminate the job immediately.',
       'How many total attempts does the background worker permit?','en'),
      ('invoice','Invoices must be submitted by the fifth business day of the month. Payments are made within 20 calendar days of invoice approval.',
       'When must an invoice be submitted, rather than paid?','en'),
      ('port','The public API listens on port 8443. The administrative API listens on port 9443 and is accessible only from the internal network.',
       'Какой порт используется для административного API во внутренней сети?','cross_ru_en'),
      ('cross-policy','В компании Берег расходы на гостиницу возмещают до 6000 рублей за ночь. На питание разрешено тратить до 1500 рублей в сутки.',
       'What is the nightly hotel reimbursement limit at Bereg?','cross_en_ru'),
      ('cross-pay','В компании Берег расходы на питание возмещают до 1500 рублей в сутки. Проезд до аэропорта оплачивают отдельно без этого ограничения.',
       'Which daily cap applies to meals at Bereg, excluding airport transport?','cross_en_ru'),
    ]
    for slug,text,query,group in controls:
        output['documents'].append({'id':'control/'+slug,'text':text,'group':group})
        output['queries'].append({'id':'control/'+slug,'text':query,'relevant_ids':['control/'+slug],'group':group})
    next(q for q in output['queries'] if q['id']=='control/cross-pay')['relevant_ids'].append('control/cross-policy')
    (ROOT/'corpus.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'documents':len(output['documents']),'queries':len(output['queries']),'groups':dict(Counter(q['group'] for q in output['queries']))}))


if __name__=='__main__':
    main()
