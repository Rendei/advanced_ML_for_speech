# Продвинутое машинное обучение для обработки речи

| Работа | Папка | Запуск |
|---|---|---|
| Лаба 1. Фузирование аудио + текст (IEMOCAP, WavLM + RoBERTa, cross-attention), отчёт формируется в рантайме | [`lab_fusion/`](lab_fusion/README.md) | `cd lab_fusion && docker compose run --rm lab` |
| Лаба 2. Многозадачное обучение (MTL): эмоции + valence/arousal/dominance + пол диктора, балансировка потерь, EDA | [`lab_mtl/`](lab_mtl/README.md) | `cd lab_mtl && docker compose run --rm lab` |
| Лаба 3. RL: Connect-4, AlphaZero-lite (свой батчевый MCTS) против DQN, оценка по идеальному солверу, Elo | [`lab_rl/`](lab_rl/README.md) | `cd lab_rl && docker compose run --rm lab` |

<<<<<<< HEAD
Каждая лаба самодостаточна: свой Dockerfile, данные и модели скачиваются внутри контейнера, отчёт (HTML) строится в рантайме.

`_archive/lab_fusion_synthetic/` — прежняя версия лабы 1 на синтетических данных (не используется, в репозиторий не входит).
=======
>>>>>>> b8406afa9f74d5402ad7fbe7b75fac0c1b6b4eeb
