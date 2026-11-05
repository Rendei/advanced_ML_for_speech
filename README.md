# Продвинутое машинное обучение для обработки речи

| Работа | Папка | Запуск |
|---|---|---|
| Лаба 1. Фузирование аудио + текст (IEMOCAP, WavLM + RoBERTa, cross-attention), отчёт формируется в рантайме | [`lab_fusion/`](lab_fusion/README.md) | `cd lab_fusion && docker compose run --rm lab` |
| Лаба 2. Многозадачное обучение (MTL): эмоции + valence/arousal/dominance + пол диктора, балансировка потерь, EDA | [`lab_mtl/`](lab_mtl/README.md) | `cd lab_mtl && docker compose run --rm lab` |

Каждая лаба самодостаточна: свой Dockerfile, данные и модели скачиваются внутри контейнера, отчёт (HTML) строится в рантайме.

`_archive/lab_fusion_synthetic/` — прежняя версия лабы 1 на синтетических данных (не используется, в репозиторий не входит).
