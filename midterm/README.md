# LLM4Rec: sweet.tv TV recommender

Воспроизводимый локальный эксперимент: интервальный join, единый временной hold-out,
popularity, личная история, Personalized PageRank, implicit ALS на NumPy и rank-fusion.

**Отчёт:** [`results/results.md`](results/results.md), включая **Iteration 2** и **Iteration 3**.
Финальная стратегия по указанию пользователя (лучший private):
[`submissions/i3_context_top3.csv`](submissions/i3_context_top3.csv) — private 0.33189 / public 0.36188.
Лучший public — `i3_context_top2.csv` (0.36307); локальный лидер Iteration 3 — `i3_context_top1.csv`.
Исходный кандидат первой итерации: [`submissions/hybrid.csv`](submissions/hybrid.csv).
Код не обращается к Kaggle и ничего не отправляет. Исходные данные лежат в `data/`
и исключены из Git родительским `.gitignore`.

## Окружение

Проверено: Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, pandas 2.3.3, macOS arm64.
Все зависимости — в `requirements.txt`. Готовая библиотека implicit не требуется.
Рекомендуется отдельное виртуальное окружение; установка зависимостей:

```bash
python3 -m pip install -r requirements.txt
```

Запускайте команды из папки `midterm/`:

```bash
# Полный пересчёт join, validation, всех сеток, sensitivity и final refit:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 src/run.py --rebuild

# Независимая проверка join на двух пользователях, уравнений ALS и формата CSV:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 src/verify.py

# Генерация Markdown-отчёта из измеренных артефактов:
python3 src/report.py
```

Чтобы повторить обучение на готовом join, уберите `--rebuild`. Отдельная подготовка:
`python3 src/prepare.py`. Повторный запуск перезаписывает соответствующие локальные
артефакты. Время и численные округления могут зависеть от платформы, версии BLAS
и нагрузки; seed и порядок разрешения ties фиксированы.

## Данные

В `data/` должны находиться оригинальные CSV:

- `dataset11-30.csv`;
- `export_arh_11-20-final.csv`;
- `export_arh_21-30-final.csv`;
- `export_arh_31-42-final.csv`;
- `submission.csv`.

Train: 2020-03-09 ≤ t < 2020-05-18; validation: 2020-05-18 ≤ t < 2020-07-27.
Оценка на 1260 пользователях template. Кандидаты — программы с эфиром в целевом
окне; повторы разрешены. Секунды пересечения агрегируются по user×programme.
Основной train-cut — 60 с; sensitivity — 1, 300, 1800 с без смены truth/параметров.
MAP@5 усредняется по всем 1260 пользователям; пустой truth даёт AP=0.
После выбора стратегии final refit использует только просмотры 11–30;
расписание 31–42 — исключительно для множества кандидатов и учёта строк.

## Структура

```text
src/prepare.py    интервальный join, аудит, кеш агрегированных взаимодействий
src/models.py     ALS, co-view PPR, метрика, единое ранжирование и fallback
src/run.py        валидация, сетки, sensitivity, примеры и финальные CSV
src/verify.py     независимые проверки
src/report.py     Markdown-отчёт из результатов реальных прогонов
results/          отчёт, JSON/CSV-метрики, кеши, предсказания и истории примеров
submissions/      hybrid.csv, ppr.csv, popularity.csv, history.csv
```

Ключевые артефакты: `data_audit.json`, `experiments.json`, `metrics.csv`,
`ppr_grid.csv`, `als_grid.csv`, `verification.json`, `validation_predictions.npz`,
`validation_scores.npz`; полный fusion-grid находится в `experiments.json`.
`heavy_history.csv`, `light_history.csv` и соответствующие `*_validation.csv`
содержат полные истории выбранных пользователей с названиями и категориями.

`results/` содержит также производные кеши данных и объёмные массивы score:
они локально исключены из Git через `midterm/.gitignore`, но восстанавливаются
командами выше. Отчёт, небольшие метрики, истории, исходный код и сабмиты доступны
для последующего осознанного добавления в репозиторий. Коммиты и push не выполнялись.

Для сдачи курса сверх локальной работы нужны материалы §9 условия: PDF,
GitHub-ссылка, настоящий скриншот returned Kaggle public-score и оригинальный
AI-session export. В этой задаче сетевые публикации и отправка на Kaggle запрещены
пользователем; такой score не заявляется.

## Iteration 2: 12-недельный hold-out и расписание как признак

После самостоятельных отправок пользователь сообщил Kaggle public/private для
четырёх исходных файлов. Они зафиксированы с указанием источника в секции
**Iteration 2** отчёта. Новые параметры выбираются по MAP@5 на обучении 11–18 /
валидации 19–30, старое разбиение служит диагностикой. Это продолжение после
обратной связи Kaggle, а не полностью независимый тест.

Добавлены свежая популярность за 2/4/6 недель, число выходов и секунды эфира
целевого окна, opportunity-adjusted popularity, условная популярность внутри
каналов/категорий пользователя и новая сетка rank-fusion. Расширенное использование
расписания явно разрешено заданием Iteration 2; тестовых просмотров нет.

После воспроизведения первой итерации командами выше:

```bash
# --rebuild пересчитывает только недельные кеши Iteration 2:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 src/iteration2.py --rebuild
python3 src/iteration2_report.py
```

Для повторного обучения на готовом недельном join уберите `--rebuild`.
Отдельная подготовка кеша: `python3 src/iteration2_data.py`.
`src/report.py` при повторной генерации первой части сохраняет готовую секцию
Iteration 2; `src/iteration2_report.py` обновляет её без дублирования.

Ключевые артефакты:

- `results/iteration2/comparison.csv` — метод → MAP@5 по трём truth на обоих split;
- `results/iteration2/fusion_grid.csv` — вся новая сетка (315 конфигураций);
- `results/iteration2/experiments.json` — выбор, проверки, время, внешний Kaggle;
- `results/iteration2/data_audit.json` — точное совпадение недельного и старого join;
- `results/iteration2/{old,new}_predictions.npz` — предсказания методов;
- `results/iteration2/report.md` — отдельная копия секции отчёта;
- `submissions/i2_*.csv` — шесть новых файлов для самостоятельной отправки.

Основной новый кандидат по валидации — `submissions/i2_best_hybrid.csv`.
Остальные: `i2_highpop_hybrid.csv`, `i2_highpop_schedule_hybrid.csv`,
`i2_pop_schedule.csv`, `i2_recent_pop.csv`, `i2_context_pop.csv`.
Старые четыре файла сохраняются. Формат каждого нового CSV, единый cold fallback,
отличие от старых сабмитов и отсутствие дублирующихся списков проверяются в прогоне.
Код по-прежнему ничего не отправляет и не использует Kaggle-score в подборе весов.

Результаты последующих отправок Iteration 2 сохранены в
`results/iteration2/kaggle_reported.json` и разделе 20 отчёта с указанием источника.
Ранее выбранный по validation `i2_best_hybrid` уступил `i2_context_pop` на Kaggle.

## Iteration 3: финальная контекстная итерация

**Направление выбрано по public score Kaggle**, полученному после Iteration 2.
Внутренние параметры и файлы выбираются только по MAP@5 top-10 на 19–30;
результаты 21–30 — диагностические. Сами Kaggle public/private не используются
в оптимизации коэффициентов. Это адаптивная работа после обратной связи, а не
независимый тест нового семейства.

Проверяются девять основ: канал по времени/числу исходных записей, категория,
50/50 rank fusion канала и категории, их произведение, совместная условная
популярность channel×category. На каждую основу — шесть коэффициентов global
smoothing и малые rank-fusion добавки history/PPR/ALS с суммарным весом ≤20%.
Всего 1134 конфигурации на каждом из двух hold-out.

После подготовки артефактов Iteration 1–2:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 src/iteration3.py --rebuild-counts
python3 src/iteration2_report.py
python3 src/iteration3_report.py
```

Без `--rebuild-counts` используется готовый count-кеш. Один просмотр для
count-профиля — одна оригинальная строка лога, не каждое пересечение с программой
или недельный фрагмент. Готовая секция Iteration 3 сохраняется при повторной
генерации более ранних частей отчёта.

Результаты — в `results/iteration3/`: `comparison.csv` (1134 варианта × два split ×
три truth), `base_grid.csv` (54 основы со сглаживанием), `experiments.json`,
`count_audit.json`, `{old,new}_predictions.npz`, `report.md`.

Выходных файлов **четыре**:

- `submissions/i3_context_pure.csv` — лучшая чистая основа без smoothing и CF;
- `submissions/i3_context_top1.csv`;
- `submissions/i3_context_top2.csv`;
- `submissions/i3_context_top3.csv`.

Три альтернативы — лидеры новой валидации с исключением точных дублей. Все четыре
файла переобучены на 11–30 и прошли read-back проверку формата/кандидатов/cold
fallback. Прежние десять файлов сохраняются. Скрипт ничего не отправляет на Kaggle.

Последующие Kaggle-результаты Iteration 3 записаны в
`results/iteration3/kaggle_reported.json` и разделе 28 отчёта. Раздел 7 включает
validation и финальные примеры `top3`/`pure` для исходных heavy/light пользователей.
Обновить отображение сохранённых результатов без переобучения:

```bash
python3 src/iteration3_report.py
python3 src/report.py
```
