"""Render results.md strictly from measured run artifacts."""
import json

import numpy as np
import pandas as pd

from prepare import DATA, OUT
from models import ap5, truths


def table(headers, rows):
    def clean(v):
        return str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(map(clean, headers))+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(clean, row))+' |' for row in rows])+'\n'


def main():
    audit = json.loads((OUT/'data_audit.json').read_text())
    exp = json.loads((OUT/'experiments.json').read_text())
    verification = json.loads((OUT/'verification.json').read_text())
    meta = {int(r['tv_show_id']): r for r in json.loads((OUT/'metadata.json').read_text())}
    train = pd.read_csv(OUT/'train_interactions.csv.gz', dtype={'user_id': str})
    val = pd.read_csv(OUT/'validation_interactions.csv.gz', dtype={'user_id': str})
    predictions = np.load(OUT/'validation_predictions.npz')
    targets = predictions['user_ids'].tolist()
    scores = np.load(OUT/'validation_scores.npz')
    items = scores['items']
    it = {int(v): i for i, v in enumerate(items)}
    truth = truths(val, targets, items)
    raw_history = {u: set(g.tv_show_id) for u, g in train.groupby('user_id')}
    eligible = train[train.seconds >= 60]
    warm = set(eligible.user_id)
    names = ['popularity', 'history', 'ppr', 'als', 'hybrid']
    labels = {'popularity': 'Popularity', 'history': 'Личная история', 'ppr': 'PPR', 'als': 'ALS', 'hybrid': 'Гибрид'}
    extra = []
    for name in names:
        pred = predictions[name]
        aps = np.array([ap5([it[int(i)] for i in row], tr) for row, tr in zip(pred, truth['top10'])])
        iswarm = np.array([u in warm for u in targets])
        familiar = sum(int(i) in raw_history.get(u, set()) for u, row in zip(targets, pred) for i in row)
        new_hits = sum(int(i) not in raw_history.get(u, set()) and it[int(i)] in tr
                       for u, row, tr in zip(targets, pred, truth['top10']) for i in row)
        extra.append([labels[name], f'{aps[iswarm].mean():.6f}', f'{aps[~iswarm].mean():.6f}',
                      len(np.unique(pred)), f'{100*familiar/pred.size:.2f}%', new_hits])
    for name in names:
        assert np.array_equal(predictions[name][~iswarm], predictions['popularity'][~iswarm])
    lines = []
    def add(s=''):
        lines.append(s+'\n')
    add('# sweet.tv — сравнение рекомендаций LLM4Rec')
    if (OUT/'iteration2'/'report.md').exists():
        add('Разделы 1–12 — сохранённая первая итерация. Обновлённые кандидаты, '
            'сообщённые пользователем Kaggle-score и сравнение двух разбиений — в секциях **Iteration 2** и **Iteration 3** ниже.')
    add('## 1. Итог и выбранная стратегия')
    add(f"Основной кандидат — **`submissions/hybrid.csv`**: MAP@5 на truth top-10 = **{exp['metrics']['hybrid']['top10']:.6f}**. "
        f"Лучший одиночный метод — PPR: **{exp['metrics']['ppr']['top10']:.6f}**. "
        f"Popularity: **{exp['metrics']['popularity']['top10']:.6f}**, личная история: **{exp['metrics']['history']['top10']:.6f}**. "
        'Реализованы два требуемых алгоритма сверх бейзлайнов: Personalized PageRank на co-view графе и implicit ALS на NumPy.')
    delta = exp['metrics']['hybrid']['top10']-exp['metrics']['ppr']['top10']
    ci = exp['hybrid_minus_best_single_bootstrap95']
    add(f"Гибрид выше PPR на {delta:.6f} абсолютных единиц ({100*delta/exp['metrics']['ppr']['top10']:.2f}% относительно PPR). "
        f"Парный bootstrap по пользователям, 2000 выборок, seed={exp['seed']}: 95% интервал разницы [{ci[0]:.6f}; {ci[1]:.6f}]. "
        'Это описательная оценка на том же hold-out, на котором выбирались параметры, а не независимое подтверждение выигрыша. '
        'На более строгом truth top-5 PPR лучше гибрида; выбор относится к заранее заданной основной цели top-10.')
    add('## 2. Окружение, воспроизводимость и происхождение')
    env = exp['environment']
    add(f"Python `{env['python'].replace(chr(10), ' ')}`; `{env['platform']}`; CPU: {env['cpu_count']} логических ядер. "
        + '; '.join(f"{k} {v}" for k, v in env['packages'].items())+'. '
        'NumPy использует системный macOS Accelerate BLAS/LAPACK (проверено через `np.show_config()`). '
        f"Seed: **{exp['seed']}**. Потоки: "+', '.join(f'`{k}={v}`' for k,v in env['threads'].items())+'.')
    add('Код написан для этой задачи с помощью AI-ассистента; готовые внешние рекомендательные реализации, внешние данные, '
        'предобученные модели и API не использовались. NumPy — линейная алгебра ALS, SciPy — sparse-матрицы графа, pandas — чтение/агрегации. '
        'Использованы только пять локальных CSV. Команды — в `../README.md`. Все числовые результаты взяты из выполненных прогонов '
        '`data_audit.json`, `experiments.json`, `verification.json` и сохранённых предсказаний. '
        'В раннем диагностическом прогоне Accelerate выдавал floating-point warnings при `matmul` с конечными результатами. '
        'В окончательном коде произведения ALS вычисляются через `np.einsum`, есть проверки конечности факторов; '
        'финальный полный прогон выполнен без этих предупреждений. Его метрики совпали с диагностическим.')
    add('## 3. Данные и интервальный join')
    add(table(['Файл', 'Строк', 'Строк id≠0', 'Доля строк id≠0', 'Уникальных id≠0', 'Каналов'],
        [[n, v['rows'], v['labelled_rows'], f"{100*v['labelled_rows']/v['rows']:.3f}%", v['programmes'], v['channels']]
         for n, v in audit['schedule'].items()]))
    v = audit['views']
    add(f"Лог: **{v['rows']}** строк, **{v['users']}** пользователей. Template: {len(targets)} уникальных пользователей; "
        'идентификаторы пользователей читаются строками: часть не помещается в signed int64. '
        f"Некорректных интервалов stop≤start: {v['nonpositive_intervals']}; вне недель 11–30: {v['outside_window_rows']}. "
        f"В {v['duration_mismatch']} строках `duraton` не равно stop−start; источником истины являются временные границы.")
    add(table(['Квантиль длительности просмотра', 'Секунд'], [[k, f'{vv:.2f}'] for k,vv in v['duration_quantiles_seconds'].items()]))
    add('### Правило сопоставления')
    add('Одинаковый канал: `vsetv_id = channel_id`. Интервал просмотра `[start_time, stop_time]` пересекается '
        'с `[start, start+duration)`; вес пересечения — `max(0, min(stop_time, start+duration) − max(start_time, start))`. '
        'В непрерывных секундах включение правой границы просмотра не меняет длительность; касание границы даёт ноль. '
        'Перед join объединены расписания 11–20 и 21–30, удаляются дубли по `(channel_id, start, end, tv_show_id)`. '
        f"Фактически удалено **{audit['schedule_duplicates_removed']}**; строк с неположительной длительностью эфира исключено **{audit['schedule_nonpositive_duration']}**. "
        'Интервалы, пересекающие границу split, разрезаются по ней; каждая секунда попадает в соответствующее окно. '
        'Используются наивные локальные даты без выдуманного timezone shift. Лог читается чанками по 250000 строк; '
        'расписание индексируется по каналу, префиксу максимального конца и `searchsorted`. '
        'Сумма секунд всех положительных пересечений агрегируется в одну запись пользователь×программа за окно.')
    add('`tv_show_id=0` — не айтем, не пользовательская фича и не рекомендация. Такие пересечения считаются только в аудите. '
        'Покрытие расписанием и доли размеченных/нулевых секунд вычислены по **объединению интервалов** каждого канала, '
        'поэтому перекрывающиеся строки расписания не удваивают знаменатель. Секунды просмотра — сумма длительностей записей лога; '
        'параллельные устройства и повторяющиеся лог-записи не объединяются в уникальное время человека.')
    add(table(['Аудит', 'Обучение 11–20', 'Валидация 21–30'],
        [[label, audit['windows']['train'][key], audit['windows']['validation'][key]] for label,key in [
            ('Строк просмотров после обрезки', 'view_rows_clipped'), ('Секунд просмотра', 'view_seconds'),
            ('Покрыто расписанием, секунд', 'covered_seconds'), ('Пересечение id≠0, секунд', 'labelled_seconds'),
            ('Пересечение id=0, секунд', 'zero_seconds'), ('Положительных пар просмотр×эфир', 'intersection_pairs'),
            ('Из них id≠0', 'labelled_pairs'), ('Пользователь×айтем до порога', 'user_item_pairs'),
            ('Пользователей с размеченной историей', 'users_with_labelled_history')]]))
    add(f"Границу split пересекли {sum(w['view_rows_clipped'] for w in audit['windows'].values())-v['rows']} строк лога: "
        'они учтены в обоих окнах разными, неперекрывающимися отрезками времени.')
    coverage_rows = []
    for key in ['train', 'validation']:
        w = audit['windows'][key]
        coverage_rows.append([key, f"{100*w['covered_seconds']/w['view_seconds']:.3f}%",
            f"{100*w['labelled_seconds']/w['view_seconds']:.3f}%", f"{100*w['zero_seconds']/w['view_seconds']:.3f}%",
            f"{100*(1-w['covered_seconds']/w['view_seconds']):.3f}%", f"{100*audit['airtime'][key]['labelled_fraction']:.3f}%"])
    add(table(['Окно', 'Покрыто расписанием', 'На id≠0 / все просмотры', 'На id=0 / все просмотры', 'Не покрыто', 'Размечено эфирных секунд по каналам'], coverage_rows))
    total_seconds = sum(w['view_seconds'] for w in audit['windows'].values())
    zero_seconds = sum(w['zero_seconds'] for w in audit['windows'].values())
    add(f"За весь лог доля просмотренных секунд на id=0: **{100*zero_seconds/total_seconds:.3f}%** ({zero_seconds}/{total_seconds}). "
        'Доля размеченных **строк** около 10.8% в первом файле — не доля эфирных секунд и не доля просмотренных секунд. '
        'Зрители непропорционально часто смотрят размеченные передачи. '
        'В обоих окнах сумма секунд размеченных пересечений совпала с union-покрытием id≠0: '
        + str(all(w['interaction_seconds_raw']==w['labelled_seconds'] for w in audit['windows'].values()))+'.')
    add('### Порог «посмотрел»')
    add(f"Основной порог фиксирован **60 секунд суммарно на пользователь×программа за окно обучения**. "
        f"После него остаётся {len(eligible)} пар (из {len(train)}). Это абсолютный cut, не процент длительности передачи. "
        'Он отсекает короткие пересечения границ эфира, но сохраняет краткие информационные передачи и повторные короткие заходы. '
        f"Медиана длительности исходной записи — {v['duration_quantiles_seconds']['0.5']:.0f} с; "
        f"медиана агрегированных положительных пар обучения — {audit['windows']['train']['positive_pair_seconds_quantiles']['0.5']:.0f} с. "
        'Порог применяется до преобразования весов одинаково во всех моделях. Truth фиксирован независимо от обучающего cut: '
        'top-10/top-5 по всем положительным секундам и отдельный вариант ≥1800 с. Это позволяет проверять чувствительность обучения без подмены цели.')
    add('## 4. Зафиксированный протокол валидации')
    add('- Train: `[2020-03-09 00:00:00, 2020-05-18 00:00:00)` (недели 11–20); validation: '
        '`[2020-05-18 00:00:00, 2020-07-27 00:00:00)` (21–30). Все модели обучены на взаимодействиях **всех** пользователей train; '
        'оценка — только на пользователях template, в исходном порядке.')
    add(f"- Кандидаты валидации: **{audit['validation_candidates']}** id≠0, эфир которых пересекает окно validation. "
        'Поэтому их количество не обязано равняться числу уникальных id в одном CSV: слот из предыдущего файла может пересечь границу. '
        'Повторы уже просмотренных программ разрешены одинаково для всех методов. '
        f"Финальное окно: `[2020-07-27, 2020-10-19)` (31–42); кандидаты из `export_arh_31-42-final.csv`: **{audit['final_candidates']}**. "
        'Расписание теста читается только для этого множества и учёта строк; его названия, жанры, актёры и частоты эфира не входят в модели.')
    add('- Основная релевантность: десять программ пользователя с наибольшими суммарными секундами в validation; '
        'дополнительная: top-5 и все программы с ≥1800 с. При равенстве секунд — меньший id. '
        'Ранжирование truth нужно только для выбора множества: внутри AP релевантность бинарная, не duration-weighted.')
    add('- `AP@5 = Σ_{r=1..5}(P@r × rel_r) / min(5, |truth|)`; для пустого truth AP=0. '
        'MAP — среднее по **всем 1260** пользователям, без исключения холодных или пустых. '
        + '; '.join(f"непустых {k}: {v}" for k,v in exp['truth_nonempty_users'].items())+'.')
    add(f"- Холодных пользователей при пороге 60 с: **{exp['validation_cold_users']}**. Единый fallback — общий popularity train, "
        'ограниченный текущими кандидатами. Если модель даёт меньше пяти положительных конечных score, остальные позиции заполняются '
        'тем же popularity без повторов. Равные score — по возрастанию id. Результат — ровно пять уникальных ненулевых id. '
        'Для холодных пользователей все пять методов дали идентичные списки (проверено по сохранённым предсказаниям).')
    add('- Только train-секунды входят в веса/граф/факторы. Validation-секунды используются для truth и выбора параметров, '
        'не для fitting в валидационном эксперименте. На этапе финального refit недели 21–30 включаются в обучение. '
        'Общий массив id — лишь индекс для выравнивания; у будущих/неизвестных обучению id нет взаимодействий и обученного сигнала. '
        'Названия и категории для таблиц берутся только из расписаний 11–30 и не используются алгоритмами.')
    add('## 5. Методы и гиперпараметры')
    add('**Popularity:** сумма секунд программы по всем train-пользователям после порога. '
        '**Личная история:** собственные суммарные train-секунды, затем общий fallback. '
        'Это сильный repeat-aware бейзлайн для сериалов и регулярно выходящих передач; он не ищет похожих пользователей.')
    add('**PPR:** взвешенная матрица `L_ui=log1p(seconds_ui/60)` нормируется L2 по пользователям; '
        '`G=L.T@L`, диагональ обнуляется, затем cosine-нормировка по столбцам L. '
        'Для каждого айтема сохраняются 50 сильнейших исходящих соседей (ties по id), после чего строки графа нормируются к вероятностям. '
        'Изолированным вершинам добавлена петля. Персональный restart `r_u` пропорционален исходным секундам собственной истории, '
        'а не валидационным просмотрам. Итерации `p=(1−alpha)r + alpha P.T p`; alpha — вероятность продолжить прогулку, '
        'restart probability — `1−alpha`. Stop: максимальное по пользователям L1-изменение <1e−6, максимум 200 итераций. '
        f"Граф: {exp['ppr_graph']['nodes']} вершин (включая изолированные id индекса), {exp['ppr_graph']['edges']} рёбер с петлями.")
    add(table(['alpha', 'MAP top-10', 'MAP top-5', 'MAP ≥1800', 'Итерации', 'L1 residual', 'Секунд'],
        [[r['alpha'], f"{r['top10']:.6f}", f"{r['top5']:.6f}", f"{r['ge1800']:.6f}", r['iterations'],
          f"{r['max_user_l1_residual']:.2e}", f"{r['seconds']:.3f}"] for r in exp['ppr_grid']]))
    add('**Implicit ALS:** минимизируется `Σ_ui c_ui(p_ui−x_u·y_i)^2 + λ(Σ||x_u||²+Σ||y_i||²)`. '
        '`p_ui=1` для пар после cut, иначе 0; ненаблюдаемые пары имеют `c_ui=1`. '
        'Для просмотренных `c_ui=1+scale*f(seconds_ui)`, где `f=log1p(seconds/60)` либо `seconds/3600`. '
        'Обновления — точные normal equations через `numpy.linalg.solve`; отсутствующие пары учтены через полный Gram. '
        '15 полных итераций user/item updates, λ=10, начальные факторы `Normal(0,0.01)`, фиксированный seed. '
        'Пользователь/айтем без взаимодействий получает нулевой фактор. Score — dot product; это не вероятность. '
        'Отрицательные score не используются для заполнения рекомендаций: действует единая положительность + fallback.')
    add(table(['Факторы', 'scale', 'transform', 'λ', 'Итерации', 'MAP top-10', 'MAP top-5', 'MAP ≥1800', 'Секунд'],
        [[r['factors'],r['scale'],r['transform'],r['regularization'],r['iterations'],f"{r['top10']:.6f}",f"{r['top5']:.6f}",f"{r['ge1800']:.6f}",f"{r['seconds']:.3f}"] for r in exp['als_grid']]))
    add(f"Выбраны PPR alpha={exp['selected_ppr_alpha']} и ALS `{exp['selected_als']}` по основной MAP@5 top-10. "
        'Content-based и association rules не реализованы и не оценены; численные сравнения с ними не заявляются.')
    add('### Гибрид: reciprocal-rank fusion')
    add('Каждый компонент сначала формирует top-100 по общему правилу с popularity-добивкой. '
        'Его вклад в программу на позиции r — `1/(20+r)` (r начинается с 1), за пределами top-100 — 0. '
        'Итог — взвешенная сумма четырёх rank-score: история, PPR, ALS, popularity. '
        'Сырые секунды, вероятности прогулки и dot products напрямую не складываются. '
        'Сетка: веса первых трёх на симплексе с шагом 0.25, умноженные на `(1−w_pop)`; '
        '`w_pop ∈ {0, 0.02, 0.05, 0.1}`. '
        f"Всего **{len(exp['fusion_grid'])}** комбинаций. Выбраны **{exp['selected_fusion_weights']}** по MAP top-10. "
        'На одинаковых результатах сохраняется первая комбинация сетки; ни финальная тестовая активность, ни Kaggle-score не использованы.')
    top_fusions = sorted(exp['fusion_grid'], key=lambda r: -r['top10'])[:10]
    add(table(['Веса history / PPR / ALS / pop', 'MAP top-10', 'MAP top-5', 'MAP ≥1800'],
        [[str(r['weights']),f"{r['top10']:.6f}",f"{r['top5']:.6f}",f"{r['ge1800']:.6f}"] for r in top_fusions]))
    add('Полная сетка fusion сохранена в `experiments.json`; остальные сетки также в `ppr_grid.csv`, `als_grid.csv`. '
        'ALS слабее popularity отдельно, но может добавлять другой порядок кандидатов в ансамбле; '
        'наличие ненулевого веса само по себе не доказывает причинную пользу компонента. '
        'Сетка содержит решения без каждого из персональных компонентов и без popularity.')
    add('## 6. Сопоставление на одном hold-out')
    add(table(['Метод', 'MAP@5 top-10 (основная)', 'MAP@5 top-5', 'MAP@5 ≥1800 с'],
        [[labels[n]]+[f"{exp['metrics'][n][k]:.6f}" for k in ['top10','top5','ge1800']] for n in names]))
    add(table(['Метод', 'MAP warm top-10', 'MAP cold top-10', 'Уникальных рекомендаций', 'Доля id из любой train-истории', 'Новые для train id, попавшие в truth top-10 (слоты)'], extra))
    add('«Новый» здесь означает только отсутствие id в размеченной train-истории данного пользователя, даже если там менее 60 секунд. '
        'Это не утверждение, что пользователь никогда не видел программу. Более высокая MAP на ≥1800 не сравнима напрямую с MAP top-10: '
        'это более широкое множество релевантности, а не улучшение модели.')
    add('### Проверка чувствительности порога')
    sensitivity = [{'threshold': 60, 'cold_users': exp['validation_cold_users'], 'train_pairs': len(eligible), 'metrics': exp['metrics']}]+exp['sensitivity']
    add(table(['Порог с', 'Пар train', 'Cold', 'Popularity', 'История', 'PPR', 'ALS', 'Гибрид'],
        [[r['threshold'],r['train_pairs'],r['cold_users']]+[f"{r['metrics'][n]['top10']:.6f}" for n in names] for r in sorted(sensitivity,key=lambda r:r['threshold'])]))
    add('Во всех sensitivity-прогонах модели заново обучены при новом cut, но **гиперпараметры, веса fusion, кандидаты, truth и пользователи не менялись**. '
        'Полные три метрики каждого метода для каждого cut — в `experiments.json`. '
        'Рост порога особенно помогает ALS; выигрыш 300/1800 у гибрида невелик. Основной протокол и финальные CSV сохраняют 60 с: '
        'sensitivity не превращена задним числом в новый основной эксперимент. Для последующего выбора cut нужна вложенная или новая временная валидация.')
    add('## 7. Два пользователя: история, рекомендации, фактическое поведение')
    from final_examples import load_examples
    final_examples = load_examples()
    add('Выбор сделан без просмотра успеха рекомендаций: 90-й и 10-й процентили суммы размеченных train-секунд среди template-пользователей '
        'с пригодной train-историей и непустым validation truth. «Тяжёлый» и «лёгкий» — относительная интенсивность **размеченного** просмотра '
        'на этой платформе. Ниже одни и те же пользователи для всех методов. В общую таблицу добавлены '
        'сохранённые validation top-5 Iteration 3; финальные списки 31–42 показаны отдельно. '
        'Название и категория — представительная запись расписания 11–30 для id (первая после сортировки по каналу и времени). '
        'Один id может объединять разные сезоны/эпизоды с разными заголовками, поэтому конкретный номер серии в таблице не означает рекомендацию именно этого эпизода.')
    for example in exp['examples']:
        label = example['label']
        add(f"### {label}: пользователь `{example['user_id']}`")
        add(f"Train: **{example['training_seconds']} с** ({example['training_seconds']/3600:.2f} ч), "
            f"{example['training_items']} положительных программ до cut; validation: {example['validation_seconds']} с. "
            f"Правило выбора: квантиль {example['quantile']}.")
        for key, title in [('history_top10','История обучения: top-10'),('actual_top10','Что смотрел в validation: top-10 (truth)')]:
            add('#### '+title)
            add(table(['id','Название','Категория','Секунд'], [[r['tv_show_id'],meta[r['tv_show_id']]['tv_show_title'],
                meta[r['tv_show_id']]['tv_show_category'],r['seconds']] for r in example[key]]))
        # Enrich complete histories with display metadata, never used for scoring.
        for suffix in ['history','validation']:
            frame = pd.read_csv(OUT/f'{label}_{suffix}.csv', dtype={'user_id':str})
            frame['title'] = frame.tv_show_id.map(lambda i:meta[i]['tv_show_title'])
            frame['category'] = frame.tv_show_id.map(lambda i:meta[i]['tv_show_category'])
            frame.to_csv(OUT/f'{label}_{suffix}.csv', index=False)
        add(f"Полные списки с секундами, названиями и категориями: `{label}_history.csv`, `{label}_validation.csv`.")
        rows = []
        history_ids = raw_history[example['user_id']]
        actual = pd.read_csv(OUT/f'{label}_validation.csv')
        actual_seconds = dict(zip(actual.tv_show_id,actual.seconds))
        for name in names:
            method = example['methods'][name]
            for pos, item in enumerate(method['ids'],1):
                rows.append([labels[name],pos,item,meta[item]['tv_show_title'],meta[item]['tv_show_category'],
                    'да' if item in history_ids else 'нет','да' if item in method['hits_top10'] else 'нет',actual_seconds.get(item,0)])
        for method,record in final_examples.get(example['user_id'],{}).items():
            truth_ids = {int(r['tv_show_id']) for r in example['actual_top10']}
            for pos,item in enumerate(record['validation'],1):
                rows.append([method,pos,item,meta[item]['tv_show_title'],meta[item]['tv_show_category'],
                    'да' if item in history_ids else 'нет','да' if item in truth_ids else 'нет',actual_seconds.get(item,0)])
        add(table(['Метод','Ранг','id','Название','Категория','Есть в train','Hit top-10','Validation с'],rows))
        add(table(['Метод','AP@5 top-10','Попадания id в truth','Знакомых / 5'],
            [[labels[n],f"{example['methods'][n]['ap5']:.6f}",' '.join(map(str,example['methods'][n]['hits_top10'])) or '—',
               f"{len(example['methods'][n]['familiar_ids'])}/5"] for n in names]))
        if example['user_id'] in final_examples:
            add('#### Финальные Top-5 на 31–42 (после обучения на 11–30)')
            final_rows = []
            for method,record in final_examples[example['user_id']].items():
                for pos,item in enumerate(record['final'],1):
                    metadata = record['final_metadata'][item]
                    final_rows.append([method,pos,item,metadata['title'],metadata['category']])
            add(table(['Стратегия','Ранг','id','Название','Категория'],final_rows))
            add('Финальные id взяты из сохранённых CSV; названия/категории — из расписания 31–42 '
                'только для отображения. Hit для тестового окна неизвестен: индивидуальных тестовых просмотров нет. '
                'Разбор ниже относится к исходным пяти методам первой итерации.')
        if label == 'heavy':
            add('**Разбор.** В train доминировали регулярные передачи, но в validation верх списка сместился к фильмам. '
                'Все рекомендации всех методов знакомы по train. Ни popularity, ни ALS не попали в top-10; '
                'история, PPR и гибрид нашли только `10002514`. История ставит его выше, поэтому для этого зрителя она лучше общего победителя. '
                'Диффузия PPR и fusion подняли общие регулярные передачи и опустили единственное попадание. '
                'Это конкретный пример нестационарности вкуса/эфира и локального ухудшения от ансамбля; '
                'нельзя выводить индивидуальный выигрыш из средней MAP.')
        else:
            add('**Разбор.** Первый по train-секундам `1000559` не попал в кандидатный список validation, '
                'поэтому история начинает с `1000381`, который действительно стал самым просматриваемым в validation. '
                'История, PPR, ALS и гибрид поставили его первым и имеют одинаковую AP; popularity не попал в truth top-10. '
                'PPR усилил co-view связанного `400363`, ALS предложил два отсутствующих в train id (`10001336`, `10003102`), '
                'но они не стали попаданиями в основной truth. При этом `10001336` действительно набрал **4428 с** в validation '
                'и релевантен в дополнительной цели ≥1800 с: это наблюдаемое расширение за пределы train-истории, '
                'хотя оно не вознаграждается truth top-10. `10003102` в validation не наблюдался. '
                'Гибрид в этом примере рекомендует только знакомые id. '
                'Нулевые секунды/промах по top-10 не доказывают неприязнь: проверяется лишь зафиксированная цель и наблюдаемое окно.')
    add('## 8. Бизнес- и алгоритмический выбор (§8)')
    add('Для офлайн-цели этой задачи я выбираю **rank-fusion историю + PPR + ALS + небольшую popularity**, '
        'поскольку он лучший по основной MAP и по дополнительному truth ≥1800. Для практического запуска преимущество '
        'над PPR следует считать предварительным: bootstrap и альтернативный top-5 не дают основания обещать устойчивый рост. '
        'PPR — сильный более простой резервный вариант; история — дешёвый обязательный контроль качества. '
        'Привычные регулярные передачи важны в ТВ, поэтому исключать повторы здесь означало бы менять пользовательскую задачу.')
    add('Popularity уже находит массовые регулярные передачи без профиля и даёт одинаковый осмысленный старт холодным пользователям, '
        'но проваливается на двух выбранных пользователях по top-10. История лучше сохраняет индивидуальную длительность и повторный спрос. '
        'PPR переносит часть массы на co-view соседей и дальше по графу; при слишком большом alpha персональный restart слабее, '
        'а качество основной цели в измеренной сетке снижается. ALS сжимает совместный просмотр в низкоранговые факторы: '
        'в текущей сетке его списки существенно слабее и более массовые, чем repeat-aware история. '
        'Это результат данного обучения, не доказательство общей непригодности MF для ТВ. '
        'Ассоциативные правила могли бы явно выделять направленные связи с support/confidence/lift и редкие локальные сочетания, '
        'которые низкоранговое сглаживание теряет; здесь такие правила не строились. '
        'Content-признаки category/genre/year/director/actors могли бы связывать новые программы при отсутствии co-view сигнала, '
        'но это гипотеза дальнейшей работы: content-модель не реализована, её прирост не измерен.')
    add('На данном масштабе sparse-граф и история недороги относительно полной сетки ALS; '
        'в production можно пересчитывать граф/факторы пакетно, а онлайн хранить top-кандидатов и выполнять лёгкую rank-fusion. '
        'Измеренное время ниже относится к пакетному Python-прогону на этом компьютере, не к latency сервиса. '
        'Два примера показывают преимущественно знакомые предпочтения; один новый для train id у ALS получил '
        'наблюдаемый просмотр в дополнительной цели, но это ещё не измерение удовлетворённости или полезности открытия. '
        'Выбор смеси повышает среднее качество, но не гарантирует разнообразие или лучший опыт каждому зрителю. '
        'Онлайн A/B-тест, выручка, удержание и удовлетворённость не измерялись и не заявляются.')
    add('## 9. Время работы и проверка реализации')
    add(table(['Этап','Измеренное время, с'], [
        ['Подготовка данных и полный join',f"{audit['preparation_seconds']:.3f}"],
        ['Построение графа validation',f"{exp['timings']['ppr_graph_seconds']:.3f}"],
        ['PPR: вся сетка (без построения графа)',f"{sum(r['seconds'] for r in exp['ppr_grid']):.3f}"],
        ['ALS: вся сетка',f"{sum(r['seconds'] for r in exp['als_grid']):.3f}"],
        ['Fusion: вся сетка',f"{exp['timings']['fusion_grid_seconds']:.3f}"],
        ['Sensitivity: три дополнительных cut',f"{sum(r['seconds'] for r in exp['sensitivity']):.3f}"],
        ['Финальный refit, scoring и CSV-проверки',f"{exp['final']['seconds']:.3f}"],
        ['Весь run.py с готовым join, включая сохранение артефактов',f"{exp['experiment_seconds']:.3f}"],
        ['Независимая проверка verify.py',f"{verification['seconds']:.3f}"]]))
    add('Время PPR/ALS-сеток включает scoring, ранжирование и метрики; построение графа вынесено отдельно. '
        'Чтение CSV, I/O и состояние компьютера влияют на воспроизводимость времени. '
        f"Независимый brute-force join для обоих примеров проверил **{verification['independent_join_view_rows']}** исходных строк просмотров "
        f"и полное совпадение **{verification['independent_join_pairs']['train']}** train-пар и **{verification['independent_join_pairs']['validation']}** validation-пар. "
        f"Для маленькой независимой dense-задачи максимальный residual normal equations ALS — {verification['als_max_equation_residual']:.3e}. "
        'Также проверены объединение перекрытых интервалов, касание границ, AP@5 и детерминированная popularity-добивка. '
        'Проверка не является исчерпывающим доказательством отсутствия ошибок на всех данных; это конкретно выполненные проверки.')
    add('## 10. Финальное обучение и файлы сабмитов')
    add(f"Выбранные параметры и веса заморожены. Взаимодействия недель 11–30 объединены, "
        f"секунды одной пары суммированы до применения cut 60; после cut **{exp['final']['train_pairs']}** пар. "
        f"Кандидаты — только расписание 31–42, {audit['final_candidates']} id. Холодных целевых пользователей: **{exp['final']['cold_users']}**. "
        'Это реальное переобучение графа и ALS, а не копирование валидационных рекомендаций.')
    add(table(['Файл','Роль','Строк без заголовка','Уникальных рекомендованных id','Формат проверен'],
        [[s['file'], 'Основной' if s['file'].endswith('/hybrid.csv') else ('Лучший одиночный' if s['file'].endswith('/ppr.csv') else 'Контрольный'),
          s['rows'],s['unique_recommended_items'],s['format_verified']] for s in exp['submissions']]))
    add('Для каждого CSV независимо проверены: точные заголовки `user_id,tv_show_id`, 1260 строк + заголовок, '
        'порядок и идентификаторы пользователей из template, ровно пять уникальных целочисленных ненулевых id через одиночный пробел '
        'и принадлежность каждого id финальному кандидатному множеству. Результат: `verification.json`, status=passed. '
        '**На Kaggle файлы не отправлялись; публичного/приватного score нет. Коммиты и push не выполнялись.**')
    add('## 11. Ограничения и границы выводов')
    add('1. Основное ограничение — большая часть строк эфира не имеет id. Покрытие эфирных секунд и просмотренных секунд отличается от '
        'доли строк; все три величины приведены выше. Мы предсказываем только размеченное поведение, а не весь интерес пользователя.\n'
        '2. Один временной hold-out использован для выбора alpha, ALS и fusion. Это оптимистичная оценка выбранной стратегии; '
        'нужна следующая временная проверка или nested rolling validation. 60-комбинационная сетка fusion усиливает selection bias.\n'
        '3. Эфирная доступность известна заранее, но пользовательские часовые пояса, подписки/доступность каналов и возможность одновременно '
        'смотреть передачи не моделируются. Программа агрегирована по id, а не по эпизоду/конкретному показу.\n'
        '4. Просмотр не равен удовлетворённости; длительность может отражать фоновый телевизор. Перекрывающиеся устройства '
        'и лог-дубли не превращены в уникальное человеческое время.\n'
        '5. Неизвестные обучению программы без content-сигнала не имеют персонального пути в PPR/ALS; popularity тоже не знает их спроса. '
        'Они могут попасть только через нулевые tie/fallback, что не решает item cold-start содержательно.\n'
        '6. Более высокий threshold улучшал некоторые модели, но основная постановка не менялась. Глобальное превосходство гибрида '
        'не переносится на каждого зрителя: тяжёлому примеру лучше служила история.\n'
        '7. Не оценивались content-based, ассоциативные правила, recency decay, сезонность и channel-aware режим. '
        'Нельзя утверждать превосходство над этими альтернативами.\n'
        '8. Метрика Kaggle и фактическое тестовое поведение не наблюдались. Офлайн MAP не выдаётся за public-score или бизнес-эффект.')
    add('## 12. Соответствие условию и оставшиеся внешние материалы')
    add('§4: реализованы и отдельно оценены PPR и implicit MF; popularity обязателен и присутствует. '
        '§5: join, нулевые id, порог, split, кандидаты, повторы, холодный старт, top-5, гиперпараметры и окружение описаны. '
        '§6: стратегия и fusion выбраны на validation. §7: два одинаковых пользователя, история и top-5 каждого метода с id/названием/категорией. '
        '§8: выбор платформенной стратегии, сравнение с фактом, вычислительная стоимость и ограничения приведены. '
        'Задание пользователя требует Markdown-отчёт и локальные CSV — они созданы. '
        'Для последующей **курсовой сдачи по §9** отдельно потребуются PDF, ссылка на опубликованный репозиторий, '
        'подлинный скриншот returned Kaggle public-score после самостоятельной отправки и оригинальный немодифицированный экспорт AI-сессии. '
        'Эти внешние материалы здесь не созданы/не подменены: доступа к результату Kaggle нет, публикация и отправка не выполнялись.')
    document = '\n'.join(lines)
    # Preserve later experimental sections when reproducing the original report.
    iteration2 = OUT/'iteration2'/'report.md'
    if iteration2.exists():
        document = document.rstrip()+'\n\n'+iteration2.read_text(encoding='utf-8')
    iteration3 = OUT/'iteration3'/'report.md'
    if iteration3.exists():
        document = document.rstrip()+'\n\n'+iteration3.read_text(encoding='utf-8')
    (OUT/'results.md').write_text(document, encoding='utf-8')
    print('Wrote', OUT/'results.md')


if __name__ == '__main__':
    main()
