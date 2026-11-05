# Лаба 2. Многозадачное обучение (MTL): эмоции + valence/arousal/dominance + пол диктора

Одна модель с общим транком (фузирование аудио и текста из лабы 1) и несколькими «головами» учится сразу на нескольких задачах.
Сравниваем её с моделями, обученными на одной задаче (single-task), и проверяем, **помогает ли многозадачность** и **как лучше складывать потери**.

| Что | Чем |
|---|---|
| Данные | IEMOCAP (зеркало HF `Ar4ikov/iemocap_audio_text`): 6 877 реплик, аудио 16 кГц, транскрипт, 4 эмоции, оценки valence/arousal/dominance (1–5), пол диктора |
| Энкодеры | `microsoft/wavlm-base-plus` (аудио) и `roberta-base` (текст), заморожены |
| Задачи | **эмоция** (4 класса, основная) · **valence, arousal, dominance** (регрессия, связанные) · **пол диктора** (классификация, несвязанная) |
| Модель | общий транк (cross-attention Q=текст ↔ K/V=аудио, «bicross») + по небольшой голове на задачу |
| Режимы | 5 single-task моделей; MTL: эмоция+VAD, эмоция+пол; все задачи с весом вспомогательных 0.1 / 0.3 / 1; uncertainty weighting (Kendall 2018); PCGrad (Yu 2020) |
| Протокол | leave-one-session-out (5 сессий, дикторы в тесте не встречаются в обучении), 3 seed'а |
| Метрики | эмоция — UA/WA/F1; VAD — CCC; пол — UA. Разница режимов — парный bootstrap **по диалогам** (95 % интервал) |
| Анализ | EDA, веса задач (uncertainty), конфликты градиентов в транке (косинусы), чувствительность к весу aux-задач, вспомогательные задачи — выигрывают ли они |

## Запуск (Docker)

Всё — загрузка данных и моделей, обучение, отчёт — выполняется **внутри контейнера**; на хосте нужен только Docker.
Первый запуск скачивает датасет (~1.4 ГБ) и модели (~1 ГБ) в `./cache` (монтируется с хоста) и один раз извлекает признаки;
дальше всё берётся из кэша. Нужен интернет (Hugging Face). Для GPU — Docker Desktop с WSL2 и драйвер NVIDIA.

```bash
docker compose run --rm eda       # EDA данных -> outputs/eda.html (CPU, без обучения)
docker compose run --rm smoke     # 600 реплик, 2 фолда: проверка конвейера -> outputs/smoke/report.html
docker compose run --rm lab       # полный эксперимент на GPU -> outputs/report.html
docker compose run --rm lab-cpu   # то же на CPU (медленно)
docker compose run --rm test      # ruff + mypy --strict + pytest
```

Без Docker Compose: `docker build -t mtllab .` и `docker run --rm --gpus all -v "$PWD/outputs:/app/outputs" -v "$PWD/cache:/app/cache" mtllab`.

Результат: `outputs/report.html` (самодостаточный HTML на русском: EDA, постановка, схема, формы тензоров, таблицы, графики, значимость,
выводы), `outputs/eda.html`, `outputs/figures/*.png`, `outputs/metrics.json`. **Отчёт формируется в рантайме** из чисел и графиков именно этого запуска.
`python -m mtllab report` пересобирает отчёт из сохранённых результатов без переобучения.

## Структура

```
src/mtllab/
  config.py       задачи, режимы обучения (Regime), конфигурация и профили
  data.py         загрузка IEMOCAP, метки всех задач, потоковое декодирование аудио
  features.py     замороженные WavLM + RoBERTa -> последовательности признаков (кэш .npz; совместим с лабой 1)
  tensors.py      padded-батчи с таргетами всех задач, leave-one-session-out
  tasks.py        потери задач, z-оценки VAD, метрики (UA, CCC, ...)
  models.py       общий транк (фузирование) + головы задач
  balancing.py    uncertainty weighting, PCGrad, косинусы градиентов
  train.py        обучение одного режима, предсказание
  experiment.py   режимы × сессии × seed'ы
  analysis.py     таблицы, bootstrap по диалогам, конфликты градиентов
  viz.py          графики (EDA и результаты)
  report.py + report.html.j2, eda.html.j2, _base/_eda_section   HTML-отчёты
tests/            быстрые тесты на случайных тензорах (без скачивания)
```

## Лицензии

IEMOCAP распространяется USC SAIL на условиях исследовательской лицензии; зеркало на HF используется только для учебной работы.
