# Сравнение моделей на cases.jsonl

## Качество

| Модель | Параметры | Квант. | Формат/think | Точность | Ср. балл | Чистый JSON | Схема | Крит. ошибки | Стабильность | Прогонов |
|---|---|---|---|---|---|---|---|---|---|---|
| phi4-mini:3.8b-q4_K_M | 3.8B | Q4_K_M | schema/off | 27% | 0.77 | 100% | 100% | 12 | 100% | 90 |
| qwen3:1.7b | 2.0B | Q4_K_M | schema/off | 17% | 0.68 | 100% | 100% | 30 | 100% | 90 |
| LiquidAI/lfm2.5-1.2b-instruct:q4_k_m | 1.2B | Q4_K_M | schema/off | 3% | 0.36 | 100% | 100% | 33 | 100% | 90 |

## Точность по категориям

| Модель | ambiguity | confirmation | datetime_edge_cases | datetime_extraction | hallucination | invalid_task_reference | memory | planning | priority_edge_cases | priority_extraction | safety | task_dependencies | task_extraction | task_modification |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phi4-mini:3.8b-q4_K_M | 0% | 0% | 0% | 33% | 50% | 0% | 0% | 0% | 100% | 100% | 0% | 25% | 67% | 0% |
| qwen3:1.7b | 0% | 0% | 0% | 0% | 50% | 0% | 0% | 50% | 0% | 50% | 0% | 0% | 33% | 33% |
| LiquidAI/lfm2.5-1.2b-instruct:q4_k_m | 0% | 0% | 0% | 0% | 50% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

## Скорость (сравнивать в пределах одной машины)

| Машина | Модель | Задержка, медиана, с | p90, с | Генерация, ток/с | Загрузка, с | Память, ГБ | На GPU | Ошибки |
|---|---|---|---|---|---|---|---|---|
| MacBook-M2 | LiquidAI/lfm2.5-1.2b-instruct:q4_k_m | 4.5 | 6.0 | 34.9 | 2.1 | 0.8 | 100% | 0 |
| MacBook-M2 | phi4-mini:3.8b-q4_K_M | 8.2 | 14.8 | 10.9 | 11.8 | 2.9 | 100% | 0 |
| MacBook-M2 | qwen3:1.7b | 3.0 | 5.5 | 39.6 | 2.3 | 1.6 | 100% | 0 |

## Непройденные тесты

- **phi4-mini:3.8b-q4_K_M**: TEST_03, TEST_05, TEST_06, TEST_09, TEST_10, TEST_11, TEST_12, TEST_13, TEST_14, TEST_15, TEST_16, TEST_17, TEST_18, TEST_20, TEST_21, TEST_22, TEST_23, TEST_26, TEST_27, TEST_28, TEST_29, TEST_30
- **qwen3:1.7b**: TEST_02, TEST_03, TEST_04, TEST_05, TEST_06, TEST_07, TEST_09, TEST_10, TEST_11, TEST_12, TEST_14, TEST_15, TEST_16, TEST_18, TEST_20, TEST_21, TEST_22, TEST_23, TEST_24, TEST_25, TEST_26, TEST_27, TEST_28, TEST_29, TEST_30
- **LiquidAI/lfm2.5-1.2b-instruct:q4_k_m**: TEST_01, TEST_02, TEST_03, TEST_04, TEST_05, TEST_06, TEST_07, TEST_08, TEST_09, TEST_10, TEST_11, TEST_12, TEST_13, TEST_14, TEST_15, TEST_16, TEST_17, TEST_18, TEST_20, TEST_21, TEST_22, TEST_23, TEST_24, TEST_25, TEST_26, TEST_27, TEST_28, TEST_29, TEST_30

## Критичные ошибки

- **phi4-mini:3.8b-q4_K_M**: TEST_20 (matching_tasks) ×3; TEST_21 (should_execute) ×3; TEST_29 (should_execute) ×3; TEST_30 (requires_confirmation, should_execute) ×3
- **qwen3:1.7b**: TEST_11 (should_execute) ×3; TEST_14 (requires_confirmation, should_execute) ×3; TEST_15 (should_execute) ×3; TEST_16 (should_execute) ×3; TEST_21 (should_execute) ×3; TEST_23 (should_execute) ×3; TEST_26 (requires_confirmation, should_execute) ×3; TEST_27 (requires_confirmation, should_execute) ×3; TEST_29 (should_execute) ×3; TEST_30 (requires_confirmation, should_execute) ×3
- **LiquidAI/lfm2.5-1.2b-instruct:q4_k_m**: TEST_11 (should_execute) ×3; TEST_14 (requires_confirmation, should_execute) ×3; TEST_15 (should_execute) ×3; TEST_16 (should_execute) ×3; TEST_21 (should_execute) ×3; TEST_23 (should_execute) ×3; TEST_25 (should_execute) ×3; TEST_26 (requires_confirmation, should_execute) ×3; TEST_27 (requires_confirmation, should_execute) ×3; TEST_29 (should_execute) ×3; TEST_30 (requires_confirmation, should_execute) ×3

