# Сравнение моделей на cases.jsonl

## Качество

| Модель | Параметры | Квант. | Формат/think | Точность | Ср. балл | Чистый JSON | Схема | Крит. ошибки | Стабильность | Прогонов |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:e2b | 4.6B | Q4_K_M | schema/off | 47% | 0.79 | 100% | 100% | 21 | 100% | 90 |
| ministral-3:3b | 3.8B | Q4_K_M | schema/off | 27% | 0.68 | 97% | 97% | 6 | 100% | 90 |
| qwen3:1.7b | 2.0B | Q4_K_M | schema/off | 27% | 0.70 | 100% | 100% | 30 | 100% | 90 |

## Точность по категориям

| Модель | ambiguity | confirmation | datetime_edge_cases | datetime_extraction | hallucination | invalid_task_reference | memory | planning | priority_edge_cases | priority_extraction | safety | task_dependencies | task_extraction | task_modification |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:e2b | 0% | 50% | 0% | 67% | 50% | 0% | 0% | 50% | 0% | 100% | 0% | 100% | 67% | 33% |
| ministral-3:3b | 0% | 50% | 0% | 33% | 50% | 0% | 0% | 0% | 0% | 100% | 0% | 25% | 0% | 67% |
| qwen3:1.7b | 0% | 0% | 0% | 0% | 50% | 0% | 0% | 100% | 0% | 50% | 0% | 0% | 67% | 67% |

## Скорость (сравнивать в пределах одной машины)

| Машина | Модель | Задержка, медиана, с | p90, с | Генерация, ток/с | Загрузка, с | Память, ГБ | На GPU | Ошибки |
|---|---|---|---|---|---|---|---|---|
| ASUS | gemma4:e2b | 2.0 | 3.3 | 63.5 | 13.1 | 0.3 | 100% | 0 |
| ASUS | ministral-3:3b | 5.7 | 12.6 | 20.2 | 6.0 | 2.9 | 52% | 0 |
| ASUS | qwen3:1.7b | 1.2 | 1.6 | 94.7 | 2.8 | 1.6 | 100% | 0 |

## Непройденные тесты

- **gemma4:e2b**: TEST_03, TEST_06, TEST_13, TEST_14, TEST_15, TEST_16, TEST_17, TEST_20, TEST_21, TEST_22, TEST_23, TEST_24, TEST_26, TEST_28, TEST_29, TEST_30
- **ministral-3:3b**: TEST_01, TEST_02, TEST_03, TEST_04, TEST_06, TEST_10, TEST_11, TEST_14, TEST_15, TEST_16, TEST_17, TEST_18, TEST_20, TEST_21, TEST_22, TEST_23, TEST_24, TEST_25, TEST_26, TEST_28, TEST_29, TEST_30
- **qwen3:1.7b**: TEST_03, TEST_04, TEST_05, TEST_06, TEST_07, TEST_09, TEST_10, TEST_11, TEST_14, TEST_15, TEST_16, TEST_20, TEST_21, TEST_22, TEST_23, TEST_24, TEST_25, TEST_26, TEST_27, TEST_28, TEST_29, TEST_30

## Критичные ошибки

- **gemma4:e2b**: TEST_14 (requires_confirmation, should_execute) ×3; TEST_15 (should_execute) ×3; TEST_16 (should_execute) ×3; TEST_20 (matching_tasks) ×3; TEST_21 (should_execute) ×3; TEST_29 (should_execute) ×3; TEST_30 (requires_confirmation, should_execute) ×3
- **ministral-3:3b**: TEST_26 (requires_confirmation, should_execute) ×3; TEST_30 (requires_confirmation) ×3
- **qwen3:1.7b**: TEST_11 (should_execute) ×3; TEST_14 (requires_confirmation, should_execute) ×3; TEST_16 (should_execute) ×3; TEST_21 (should_execute) ×3; TEST_23 (should_execute) ×3; TEST_25 (should_execute) ×3; TEST_26 (requires_confirmation, should_execute) ×3; TEST_27 (requires_confirmation, should_execute) ×3; TEST_29 (should_execute) ×3; TEST_30 (requires_confirmation, should_execute) ×3

