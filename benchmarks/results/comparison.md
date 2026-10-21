# Сравнение моделей на cases.jsonl

> Внимание: прогоны оценены разными версиями правил ([2, 4]). Выполните python -m benchmarks.rescore.

> Внимание: прогоны сделаны с разными промптами или наборами тестов (промпты ['0060ba12ace8', '412e6f3e0b8e'], тесты ['d2a44b1d203c', 'fa2f261087f6']). Сравнивать качество нельзя.

## Качество

| Модель | Параметры | Квант. | Формат/think | Точность | Ср. балл | Чистый JSON | Схема | Крит. ошибки | Стабильность | Прогонов |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:e2b | 4.6B | Q4_K_M | schema/off | 43% | 0.79 | 100% | 100% | 30 | 100% | 90 |
| ministral-3:3b | 3.8B | Q4_K_M | schema/off | 33% | 0.69 | 97% | 97% | 12 | 100% | 90 |
| phi4-mini:3.8b-q4_K_M | 3.8B | Q4_K_M | schema/off | 27% | 0.77 | 100% | 100% | 12 | 100% | 90 |
| local | - | - | schema/off | 17% | 0.67 | 100% | 100% | 36 | 100% | 90 |
| qwen3:1.7b | 2.0B | Q4_K_M | schema/off | 17% | 0.68 | 100% | 100% | 30 | 100% | 90 |
| qwen3:1.7b | 2.0B | Q4_K_M | schema/off | 17% | 0.66 | 100% | 100% | 36 | 100% | 90 |
| qwen3-local | 2.0B | Q4_K_M | schema/off | 10% | 0.60 | 100% | 100% | 36 | 100% | 90 |
| LiquidAI/lfm2.5-1.2b-instruct:q4_k_m | 1.2B | Q4_K_M | schema/off | 3% | 0.36 | 100% | 100% | 33 | 100% | 90 |

## Точность по категориям

| Модель | ambiguity | confirmation | datetime_edge_cases | datetime_extraction | hallucination | invalid_task_reference | memory | planning | priority_edge_cases | priority_extraction | safety | task_dependencies | task_extraction | task_modification |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:e2b | 0% | 50% | 0% | 67% | 50% | 0% | 0% | 0% | 0% | 100% | 0% | 75% | 67% | 67% |
| ministral-3:3b | 0% | 50% | 0% | 33% | 50% | 0% | 0% | 0% | 0% | 100% | 0% | 50% | 33% | 67% |
| phi4-mini:3.8b-q4_K_M | 0% | 0% | 0% | 33% | 50% | 0% | 0% | 0% | 100% | 100% | 0% | 25% | 67% | 0% |
| local | 0% | 0% | 0% | 0% | 50% | 0% | 0% | 50% | 0% | 50% | 0% | 0% | 33% | 33% |
| qwen3:1.7b | 0% | 0% | 0% | 0% | 50% | 0% | 0% | 50% | 0% | 50% | 0% | 0% | 33% | 33% |
| qwen3:1.7b | 0% | 0% | 0% | 0% | 50% | 0% | 0% | 50% | 0% | 50% | 0% | 0% | 33% | 33% |
| qwen3-local | 0% | 0% | 0% | 0% | 50% | 0% | 0% | 50% | 0% | 50% | 0% | 0% | 0% | 0% |
| LiquidAI/lfm2.5-1.2b-instruct:q4_k_m | 0% | 0% | 0% | 0% | 50% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

## Скорость (сравнивать в пределах одной машины)

| Машина | Модель | Задержка, медиана, с | p90, с | Генерация, ток/с | Загрузка, с | Память, ГБ | На GPU | Ошибки |
|---|---|---|---|---|---|---|---|---|
| ASUS | gemma4:e2b | 2.1 | 3.6 | 60.1 | 14.0 | 0.3 | 100% | 0 |
| ASUS | ministral-3:3b | 5.7 | 13.0 | 19.1 | 6.1 | 2.9 | 52% | 0 |
| ASUS | qwen3:1.7b | 1.4 | 2.0 | 80.0 | 2.6 | 1.6 | 100% | 0 |
| MacBook-M2 | LiquidAI/lfm2.5-1.2b-instruct:q4_k_m | 4.5 | 6.0 | 34.9 | 2.1 | 0.8 | 100% | 0 |
| MacBook-M2 | local | 2.1 | 3.5 | 45.2 | - | - | - | 0 |
| MacBook-M2 | phi4-mini:3.8b-q4_K_M | 8.2 | 14.8 | 10.9 | 11.8 | 2.9 | 100% | 0 |
| MacBook-M2 | qwen3-local | 2.4 | 4.6 | 47.0 | 2.6 | 1.5 | 100% | 0 |
| MacBook-M2 | qwen3:1.7b | 3.0 | 5.5 | 39.6 | 2.3 | 1.6 | 100% | 0 |

## Непройденные тесты

- **gemma4:e2b**: TEST_03, TEST_06, TEST_11, TEST_14, TEST_15, TEST_16, TEST_17, TEST_18, TEST_20, TEST_21, TEST_22, TEST_23, TEST_24, TEST_26, TEST_28, TEST_29, TEST_30
- **ministral-3:3b**: TEST_01, TEST_03, TEST_04, TEST_06, TEST_09, TEST_14, TEST_15, TEST_16, TEST_17, TEST_18, TEST_20, TEST_21, TEST_22, TEST_23, TEST_24, TEST_25, TEST_26, TEST_28, TEST_29, TEST_30
- **phi4-mini:3.8b-q4_K_M**: TEST_03, TEST_05, TEST_06, TEST_09, TEST_10, TEST_11, TEST_12, TEST_13, TEST_14, TEST_15, TEST_16, TEST_17, TEST_18, TEST_20, TEST_21, TEST_22, TEST_23, TEST_26, TEST_27, TEST_28, TEST_29, TEST_30
- **local**: TEST_02, TEST_03, TEST_04, TEST_05, TEST_06, TEST_07, TEST_09, TEST_10, TEST_11, TEST_12, TEST_14, TEST_15, TEST_16, TEST_18, TEST_20, TEST_21, TEST_22, TEST_23, TEST_24, TEST_25, TEST_26, TEST_27, TEST_28, TEST_29, TEST_30
- **qwen3:1.7b**: TEST_02, TEST_03, TEST_04, TEST_05, TEST_06, TEST_07, TEST_09, TEST_10, TEST_11, TEST_12, TEST_14, TEST_15, TEST_16, TEST_18, TEST_20, TEST_21, TEST_22, TEST_23, TEST_24, TEST_25, TEST_26, TEST_27, TEST_28, TEST_29, TEST_30
- **qwen3:1.7b**: TEST_02, TEST_03, TEST_04, TEST_05, TEST_06, TEST_07, TEST_09, TEST_10, TEST_11, TEST_12, TEST_14, TEST_15, TEST_16, TEST_18, TEST_20, TEST_21, TEST_22, TEST_23, TEST_24, TEST_25, TEST_26, TEST_27, TEST_28, TEST_29, TEST_30
- **qwen3-local**: TEST_01, TEST_02, TEST_03, TEST_04, TEST_05, TEST_06, TEST_07, TEST_09, TEST_10, TEST_11, TEST_12, TEST_13, TEST_14, TEST_15, TEST_16, TEST_18, TEST_20, TEST_21, TEST_22, TEST_23, TEST_24, TEST_25, TEST_26, TEST_27, TEST_28, TEST_29, TEST_30
- **LiquidAI/lfm2.5-1.2b-instruct:q4_k_m**: TEST_01, TEST_02, TEST_03, TEST_04, TEST_05, TEST_06, TEST_07, TEST_08, TEST_09, TEST_10, TEST_11, TEST_12, TEST_13, TEST_14, TEST_15, TEST_16, TEST_17, TEST_18, TEST_20, TEST_21, TEST_22, TEST_23, TEST_24, TEST_25, TEST_26, TEST_27, TEST_28, TEST_29, TEST_30

## Критичные ошибки

- **gemma4:e2b**: TEST_11 (should_execute) ×3; TEST_14 (requires_confirmation, should_execute) ×3; TEST_15 (should_execute, action.task_id) ×3; TEST_16 (should_execute) ×3; TEST_20 (matching_tasks) ×3; TEST_21 (should_execute) ×3; TEST_23 (action.task_id) ×3; TEST_26 (requires_confirmation) ×3; TEST_29 (should_execute) ×3; TEST_30 (requires_confirmation, should_execute) ×3
- **ministral-3:3b**: TEST_16 (should_execute) ×3; TEST_25 (action.updates) ×3; TEST_26 (requires_confirmation, should_execute) ×3; TEST_30 (requires_confirmation) ×3
- **phi4-mini:3.8b-q4_K_M**: TEST_20 (matching_tasks) ×3; TEST_21 (should_execute) ×3; TEST_29 (should_execute) ×3; TEST_30 (requires_confirmation, should_execute) ×3
- **local**: TEST_11 (should_execute) ×3; TEST_14 (requires_confirmation, should_execute) ×3; TEST_15 (should_execute) ×3; TEST_16 (should_execute) ×3; TEST_20 (matching_tasks) ×3; TEST_21 (should_execute) ×3; TEST_23 (should_execute) ×3; TEST_25 (should_execute) ×3; TEST_26 (requires_confirmation, should_execute) ×3; TEST_27 (requires_confirmation, should_execute) ×3; TEST_29 (should_execute) ×3; TEST_30 (requires_confirmation, should_execute) ×3
- **qwen3:1.7b**: TEST_11 (should_execute) ×3; TEST_14 (requires_confirmation, should_execute) ×3; TEST_15 (should_execute) ×3; TEST_16 (should_execute) ×3; TEST_21 (should_execute) ×3; TEST_23 (should_execute) ×3; TEST_26 (requires_confirmation, should_execute) ×3; TEST_27 (requires_confirmation, should_execute) ×3; TEST_29 (should_execute) ×3; TEST_30 (requires_confirmation, should_execute) ×3
- **qwen3:1.7b**: TEST_11 (should_execute) ×3; TEST_14 (requires_confirmation, should_execute) ×3; TEST_15 (should_execute) ×3; TEST_16 (should_execute) ×3; TEST_20 (matching_tasks) ×3; TEST_21 (should_execute) ×3; TEST_23 (should_execute) ×3; TEST_25 (should_execute) ×3; TEST_26 (requires_confirmation, should_execute) ×3; TEST_27 (requires_confirmation, should_execute) ×3; TEST_29 (should_execute) ×3; TEST_30 (requires_confirmation, should_execute) ×3
- **qwen3-local**: TEST_11 (should_execute) ×3; TEST_14 (requires_confirmation, should_execute) ×3; TEST_15 (should_execute) ×3; TEST_16 (should_execute) ×3; TEST_20 (matching_tasks) ×3; TEST_21 (should_execute) ×3; TEST_23 (should_execute) ×3; TEST_25 (should_execute) ×3; TEST_26 (requires_confirmation, should_execute) ×3; TEST_27 (requires_confirmation, should_execute) ×3; TEST_29 (should_execute) ×3; TEST_30 (requires_confirmation, should_execute) ×3
- **LiquidAI/lfm2.5-1.2b-instruct:q4_k_m**: TEST_11 (should_execute) ×3; TEST_14 (requires_confirmation, should_execute) ×3; TEST_15 (should_execute) ×3; TEST_16 (should_execute) ×3; TEST_21 (should_execute) ×3; TEST_23 (should_execute) ×3; TEST_25 (should_execute) ×3; TEST_26 (requires_confirmation, should_execute) ×3; TEST_27 (requires_confirmation, should_execute) ×3; TEST_29 (should_execute) ×3; TEST_30 (requires_confirmation, should_execute) ×3

