"""Record the final adaptive experiment using executed local artifacts only."""
import json

import pandas as pd

from iteration3 import I3
from prepare import OUT
from report import table

MARKER = '\n# Iteration 3\n'


def main():
    exp = json.loads((I3/'experiments.json').read_text())
    audit = json.loads((I3/'count_audit.json').read_text())
    grid = pd.read_csv(I3/'comparison.csv')
    i2 = json.loads((OUT/'iteration2'/'experiments.json').read_text())
    feedback_path = I3/'kaggle_reported.json'
    feedback = json.loads(feedback_path.read_text()) if feedback_path.exists() else None
    lines = []
    def add(text=''):
        lines.append(text+'\n')
    def metrics(row):
        return [f"{row[k]:.6f}" for k in ['old_top10','new_top10','old_top5','new_top5','old_ge1800','new_ge1800']]
    add('# Iteration 3')
    add('## 21. Последняя итерация: направление выбрано по public score Kaggle')
    add('**Семейство channel/category popularity выбрано после получения public score Kaggle:** '
        '`i2_context_pop.csv` получил **public 0.34906**, лучший среди сообщённых результатов Iteration 1–2. '
        'Пользователь также сообщил private 0.32243; этот результат записан, но не используется для поиска параметров. '
        'Полная таблица последней обратной связи — в разделе 20. Источник чисел — сообщение пользователя, '
        'не самостоятельная отправка или проверка страницы агентом. '
        'Таким образом, направление Iteration 3 — **адаптивный выбор по public-score**, а не заранее независимое семейство.')
    add('**Внутри выбранного семейства** параметры сглаживания и веса fusion выбираются только по локальной '
        '**MAP@5 top-10 на train 11–18 / validation 19–30**. Старое разбиение train 11–20 / validation 21–30 '
        'служит диагностикой и не используется даже как tie-breaker. '
        'На момент локального выбора новые Kaggle-score были неизвестны; полученная позже обратная связь — в разделе 28. '
        'Четыре выходных файла зафиксированы до следующего сообщения пользователя о score; '
        'новых циклов экспериментов в этой итерации не планируется.')
    add('## 22. Что сохранено в протоколе')
    add('Join, секунды пересечения, исключение id=0 из айтемов, дедупликация расписания, cut 60 секунд user×item '
        'за соответствующее train-окно, разрешённые повторы и кандидаты по реальному эфиру сохранены. '
        'Метрика усредняется по всем 1260 пользователям, пустой truth даёт AP=0. '
        'Top-10 — единственный критерий выбора, top-5 и ≥1800 с — дополнительные оценки. '
        'Для cold-start в каждом методе и каждой смеси используется один и тот же all-time train popularity, '
        'им же заполняются недостающие позиции. Tie — меньший id. '
        'Профили и conditional popularity строятся только до cutoff; категория id берётся из pre-cut расписания '
        'по наибольшему объёму эфира, как в Iteration 2. '
        'Будущее расписание используется для кандидатов и доступности программы на канале, '
        'но никакое будущее viewing behaviour не используется. '
        'История/PPR/ALS переобучены в каждом окне с замороженными параметрами Iteration 1; '
        'нового подбора их внутренних гиперпараметров нет. Seed=20261024, окружение — раздел 2.')
    add(table(['Окно','Пар train после cut','Кандидатов','Cold','Источник выбора'],
        [[name,v['summary']['train_pairs'],v['summary']['candidates'],v['summary']['cold_users'],
          'selection' if name=='new' else 'diagnostic'] for name,v in exp['windows'].items()]+
        [['final',exp['final']['train_pairs'],exp['final']['candidates'],exp['final']['cold_users'],'refit 11–30 → 31–42']]))
    add('Оба локальных hold-out перекрываются; исходные PPR/ALS-параметры уже выбирались по части этих дат. '
        'Поэтому результаты не выдаются за независимую оценку обобщения. '
        'Итерация 2 показала, что даже лидер обеих локальных валидаций может уступить контекстной popularity на Kaggle.')
    add('## 23. Основа: каналы, категории, произведение и совместное условие')
    add('Обозначим `C(u,i)=Σ_c q_u(c) P_train(i|c) 1[availability(i,c)]`, '
        '`G(u,i)=q_u(g_i) P_train(i|g_i)`. Item/category и conditional popularity рассчитываются по размеченным '
        'train-взаимодействиям после общего cut. Категориальный профиль всегда взвешен секундами. '
        'Два канальных профиля используют все train-просмотры, включая эфир без id:')
    add('- **time:** `q_u(c)` — доля секунд исходных интервалов на канале; как в Iteration 2.\n'
        '- **count:** `q_u(c)` — доля исходных записей просмотра на канале. '
        'Одна запись лога считается один раз независимо от числа затронутых программ или недель; '
        'её счёт относится к неделе начала (после обрезки началом общего диапазона). '
        'Это число сессий/записей в логе, не уникальные дни и не факт удовлетворённости. '
        'Запись, начавшаяся до cutoff и продолжившаяся после него, учитывается в count один раз; '
        'в time-профиле используются только секунды до cutoff.')
    add(f"Для count-профилей прочитаны все CSV-чанки, отобрано **{audit['target_view_records']}** исходных записей "
        f"целевых пользователей; кеш — **{audit['weekly_count_rows']}** user×channel×week строк. "
        'Сумма counts до каждого cutoff независимо сверена при проходе по исходному логу:')
    add(table(['Последняя train-неделя','Исходных записей просмотра целевых пользователей'],
        [[w,n] for w,n in audit['records_before_cutoffs'].items()]))
    add('Проверены **9 основ**:\n'
        '1. `time_channel`, `count_channel`: только C.\n'
        '2. `category`: только G (тип канального профиля неприменим).\n'
        '3. `time_rrf`, `count_rrf`: 50/50 rank fusion C и G, `1/(20+r)` по top-100. '
        '`time_rrf` точно воспроизводит отправленный `i2_context_pop`.\n'
        '4. `time_product`, `count_product`: поэлементное **произведение C×G** после нормировки каждого '
        'по доступным кандидатам пользователя. Это пересечение сигналов, не RRF и не утверждение независимости предпочтений.\n'
        '5. `time_joint`, `count_joint`: действительно совместная conditional popularity '
        '`Σ_c q_u(c) q_u(g_i) P_train(i | c,g_i) 1[availability(i,c)]`. '
        'Нормировка item-популярности выполняется внутри пары channel×category до availability mask. '
        'Этот вариант отличен от произведения двух готовых score.')
    pure = grid[(grid.cf_total==0)&(grid.global_smoothing==0)]
    add('### Чистые основы без global smoothing и CF-добавок')
    add(table(['Основа','21–30 top-10','19–30 top-10','21–30 top-5','19–30 top-5','21–30 ≥1800','19–30 ≥1800'],
        [[r['base']]+metrics(r) for r in pure.to_dict('records')]))
    add('## 24. Сглаживание с глобальной popularity')
    add('Каждый контекстный score сначала обнуляется вне кандидатов и нормируется по строке: '
        '`B_norm(u,i)=B(u,i)/Σ_j B(u,j)`. При нулевой строке остаётся ноль, затем действует общий fallback. '
        'Сглаживание: `(1−λ) B_norm(u,i) + λ Pop_norm(i)`, где Pop_norm — all-time train-seconds, '
        'нормированные по доступным кандидатам. λ∈{0,0.05,0.1,0.2,0.4,0.6}. '
        'Эти нормировки нужны для сопоставимости масштабов; score не объявляется вероятностью просмотра. '
        f"Всего **{exp['smoothing_configurations']}** контекстно-глобальных вариантов (включая λ=0).")
    add('Лучшее λ **отдельно для каждой основы по новой валидации**:')
    smooth = grid[grid.cf_total==0]
    best_smooth = [g.sort_values('new_top10',ascending=False,kind='stable').iloc[0].to_dict() for _,g in smooth.groupby('base',sort=False)]
    add(table(['Основа','λ (выбор по 19–30)','MAP 21–30','MAP 19–30'],
        [[r['base'],r['global_smoothing'],f"{r['old_top10']:.6f}",f"{r['new_top10']:.6f}"] for r in best_smooth]))
    add('Полная таблица всех λ на обоих разбиениях по трём truth — `iteration3/base_grid.csv`. '
        'Наличие λ=0 в сетке позволяет сохранить несглаженный профиль, если global popularity не помогает.')
    add('## 25. Малые смеси с history / PPR / ALS')
    add('После сглаживания основа переводится в rank-score top-100 `R_B=1/(20+r)`. '
        'History/PPR/ALS переводятся в ту же шкалу с общей popularity-добивкой. '
        'Смесь: `(1−Σw) R_B + w_H R_H + w_P R_P + w_A R_A`. '
        'Для одного компонента w∈{0.025,0.05,0.1,0.15,0.2}. '
        'Дополнительно проверены H/PPR=0.05/0.05 и 0.1/0.1, H/ALS=0.05/0.05, '
        'PPR/ALS=0.05/0.05, H/PPR/ALS=0.05/0.05/0.05. '
        'Суммарная доля CF не больше **20%**. '
        'На каждую основу и λ приходится 21 конфигурация, включая отсутствие CF: '
        f"**{exp['configurations']}** вариантов на каждом hold-out. "
        'При равенстве MAP выбирается первый в фиксированной сетке, без обращения к старому split или Kaggle-score.')
    family = []
    for component in ['history','ppr','als']:
        subset = grid[grid.cf_weights.map(lambda s:set(json.loads(s))=={component})]
        r = subset.sort_values('new_top10',ascending=False,kind='stable').iloc[0]
        family.append([component,r['base'],r.global_smoothing,r.cf_weights,f'{r.old_top10:.6f}',f'{r.new_top10:.6f}'])
    multi = grid[grid.cf_weights.map(lambda s:len(json.loads(s))>1)]
    r = multi.sort_values('new_top10',ascending=False,kind='stable').iloc[0]
    family.append(['combined',r['base'],r.global_smoothing,r.cf_weights,f'{r.old_top10:.6f}',f'{r.new_top10:.6f}'])
    add('Лучший представитель каждого CF-семейства, выбор по 19–30:')
    add(table(['Добавка','Основа','Global λ','CF-веса','MAP 21–30','MAP 19–30'],family))
    add('Первые десять всей сетки по новой основной метрике:')
    add(table(['id','Основа','λ','CF-веса','MAP 21–30','MAP 19–30'],
        [[r.id,r.base,r.global_smoothing,r.cf_weights,f'{r.old_top10:.6f}',f'{r.new_top10:.6f}']
         for r in grid.sort_values('new_top10',ascending=False,kind='stable').head(10).itertuples()]))
    add('## 26. Выходные файлы и сравнение с исходной контекстной основой')
    add('Экспортируется лучший **чистый** контекстный вариант (λ=0, CF=0) и три лидирующие альтернативы '
        'по MAP 19–30. Для альтернатив исключаются точные дубли уже отправленных финальных списков, '
        'дубли выбранных финальных списков и дубли их validation top-5. '
        'Это проверка разнообразия файлов по допустимым предсказаниям, не подбор по скрытому тестовому поведению. '
        'Финальное обучение использует недели 11–30; веса и формулы заморожены до refit.')
    add(table(['Файл','Основа','Global λ','CF-веса','MAP 21–30','MAP 19–30','Изменено пользователей против i2_context','Уникальных id'],
        [[s['file'],s['config']['base'],s['config']['global_smoothing'],json.dumps(s['config']['cf_weights']),
          f"{s['config']['old_top10']:.6f}",f"{s['config']['new_top10']:.6f}",s['changed_users_vs_i2_context'],s['unique_items']]
         for s in exp['selected']]))
    add('Дополнительные truth экспортированных вариантов:')
    add(table(['Файл','21–30 top-5','19–30 top-5','21–30 ≥1800','19–30 ≥1800'],
        [[s['file']]+[f"{s['config'][k]:.6f}" for k in ['old_top5','new_top5','old_ge1800','new_ge1800']]
         for s in exp['selected']]))
    add(table(['Ориентир','MAP 21–30','MAP 19–30','Public Kaggle (только известные отправки)'],[
        ['i2_context_pop',f"{exp['windows']['old']['legacy_i2_context_metrics']['top10']:.6f}",
         f"{exp['windows']['new']['legacy_i2_context_metrics']['top10']:.6f}",'0.34906'],
        ['i2_best_hybrid',f"{i2['old_metrics']['i2_best_hybrid']['top10']:.6f}",f"{i2['new_metrics']['i2_best_hybrid']['top10']:.6f}",'0.31535'],
        *[[s['file'].split('/')[-1],f"{s['config']['old_top10']:.6f}",f"{s['config']['new_top10']:.6f}",
           f"{feedback['scores'][s['file'].split('/')[-1][:-4]]['public']:.5f}" if feedback else 'неизвестен'] for s in exp['selected']]]))
    add('**Вывод:** новый чистый product-профиль и добавление 20% history улучшили локальную MAP относительно '
        'исходного `i2_context_pop`; веса каналов по времени выиграли выбор основы. '
        'При этом даже лучший новый вариант не превосходит `i2_best_hybrid` по локальной top-10. '
        'Эта итерация намеренно исследует контекстное семейство по внешнему public-сигналу, '
        'а не объявляет локальный максимум всей работы единственным кандидатом. '
        'Малый разрыв между top1/top2 не доказывает значимого различия. '
        'Чистый вариант важен как контроль: ранее более сильная локальная персонализация уже переносилась на Kaggle хуже. '
        'До отправки нельзя было заключить, что новые файлы превысят public 0.34906; '
        'последующие сообщённые результаты и окончательный выбор отделены от локального поиска в разделе 28.')
    add('## 27. Проверки, время и воспроизведение')
    add('Независимое чтение каждого выходного CSV проверило: 1260 строк + заголовок; '
        'точные столбцы `user_id,tv_show_id`; исходный порядок и уникальность пользователей; '
        'ровно пять уникальных ненулевых цифровых id через одиночный пробел; '
        'принадлежность каждого id множеству 6632 кандидатов из расписания 31–42. '
        'Все четыре файла отличаются друг от друга и от десяти прежних файлов. '
        'Cold fallback проверен для всех 1134 конфигураций на обоих split и всех экспортированных финальных списков. '
        'Переписанная контекстная основа точно воспроизвела старый `i2_context_pop` на обоих validation и финале; '
        'history/PPR/ALS точно воспроизвели сохранённые валидационные списки Iteration 2. '
        'Старые CSV, входные данные и результаты обучения прежних итераций не перезаписывались.')
    add(table(['Этап','Измеренное время, с'],[
        ['Подготовка counts и сверка сумм',f"{audit['seconds']:.3f}"],
        ['Fit компонентов 19–30',f"{exp['windows']['new']['summary']['fit_seconds']:.3f}"],
        ['1134 конфигурации на 19–30, включая метрики и сохранение предсказаний',f"{exp['windows']['new']['evaluation_seconds']:.3f}"],
        ['Fit компонентов 21–30',f"{exp['windows']['old']['summary']['fit_seconds']:.3f}"],
        ['1134 конфигурации на 21–30, включая метрики и сохранение предсказаний',f"{exp['windows']['old']['evaluation_seconds']:.3f}"],
        ['Final refit компонентов',f"{exp['final']['fit_seconds']:.3f}"],
        ['Весь iteration3.py, включая counts, I/O, выбор и CSV-проверки',f"{exp['seconds']:.3f}"]]))
    add('Артефакты: `iteration3/comparison.csv` — все 1134 варианта на обоих split по трём truth; '
        '`iteration3/base_grid.csv` — 54 основы со сглаживанием без CF; '
        '`iteration3/experiments.json` — выбранные файлы, параметры, проверки и время; '
        '`iteration3/count_audit.json` — аудит исходных записей для count-профиля; '
        '`iteration3/old_predictions.npz`, `iteration3/new_predictions.npz` — top-5 всех конфигураций. '
        'Кеш count-профилей CSV.gz исключён из Git. Команды из `midterm/`, после артефактов Iteration 1–2:')
    add('```bash\n'
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 src/iteration3.py --rebuild-counts\n'
        'python3 src/iteration2_report.py\n'
        'python3 src/iteration3_report.py\n'
        '```')
    add('Генерация секций идемпотентна: повторное создание предыдущих частей сохраняет готовую Iteration 3. '
        '**Коммиты, push и отправки на Kaggle агентом не выполнялись.** '
        'Это последняя запрошенная итерация; её полный локальный протокол и четыре файла зафиксированы.')
    if feedback:
        add('## 28. Полученные Kaggle-score и окончательная стратегия')
        add('Источник — последующее сообщение пользователя о самостоятельно выполненных late submissions; '
            'агент не отправлял файлы и не проверял страницу Kaggle. Сохранённые локальные результаты не изменены.')
        add(table(['Файл','Private MAP@5 (сообщён)','Public MAP@5 (сообщён)'],
            [[n+'.csv',f"{s['private']:.5f}",f"{s['public']:.5f}"] for n,s in feedback['scores'].items()]))
        add('**Финальная стратегия — `submissions/i3_context_top3.csv`**, по прямому указанию пользователя: '
            'максимальный сообщённый private-score **0.33189**, public **0.36188**. '
            'Это product channel×category с time-профилем каналов, global smoothing λ=0.10 '
            'и rank fusion 80% контекстной основы / 20% истории. '
            'Итоговый выбор сделан **после** получения private-score, а не по локальной валидации. '
            'Новых параметров по этим числам не подбиралось.')
        diff = feedback['scores']['i3_context_top2']['public']-feedback['scores']['i3_context_top3']['public']
        add(f'По public лидер — `i3_context_top2.csv` (**0.36307**); его преимущество над top3 равно **{diff:.5f}** '
            '(немного больше 0.001). Статистическая значимость этой разницы не установлена. '
            'Критерий окончательного выбора private указан пользователем как требование преподавателя; '
            'при этом исходный текст задания §10 описывает ранжирование по public. '
            'В отчёте сохранены оба критерия и оба score, без переобозначения метрик.')
        add('Для тех же heavy/light пользователей раздел 7 дополнен validation top-5 `i3_context_top3` и '
            '`i3_context_pure` рядом с исходными методами (id, название, категория, hit по прежнему top-10 truth), '
            'а также их финальными top-5 из CSV на 31–42. Предсказания взяты из сохранённых артефактов, '
            'переобучение или новый подбор не выполнялись.')
    section = '\n'.join(lines)
    (I3/'report.md').write_text(section,encoding='utf-8')
    path = OUT/'results.md'
    base = path.read_text().split(MARKER)[0].rstrip()
    path.write_text(base+'\n\n'+section,encoding='utf-8')
    print('Updated',path,'and',I3/'report.md')


if __name__ == '__main__':
    main()
