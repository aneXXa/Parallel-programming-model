# Лабораторная работа 3 — Task 2 (клиент-сервер, потоки)

## Что реализовано

- Два варианта **доставки результата** (см. примеры в `AddTask/`):
  1. **slot** — `std::mutex` + `std::condition_variable` + `std::optional<T>` на задачу (ручное ожидание готовности).
  2. **promise** — `std::promise<T>` + `std::shared_future<T>`: сервер вызывает `set_value` / `set_exception`, клиент ждёт через `shared_future::get()` (аналогично идее `promise`/`future` из лекции; `shared_future` допускает повторное чтение одного и того же значения, если это понадобится).
- Два варианта **контейнера ожидающих задач** (поиск по `id`):
  - `std::unordered_map` — среднее \(O(1)\);
  - `std::map` — \(O(\log n)\), упорядоченность по `id` (для данной работы обычно не нужна, но сравним по времени).
- Очередь задач: `std::queue` + `std::condition_variable` (как в `example_add_package.cpp` — ожидание без «busy sleep»).
- Интерфейс сервера (одинаковый для всех вариантов):
  - `void start()`
  - `void stop()`
  - `size_t add_task(task)`
  - `T request_result(id)` — блокирующий
- Три клиентских потока (`sin`, `sqrt`, `pow`), каждый пишет свой `.csv`.
- Программа **`task2_benchmark`**: прогоняет все 4 комбинации и пишет `task2_benchmark_results.csv`; в консоль выводится рекомендуемый флаг для основной программы.
- Тест `verify_results.cpp` проверяет корректность чисел в файлах.

Файлы реализации:

- `server_slot.hpp` — вариант «mutex+cv»
- `server_promise.hpp` — вариант «promise/shared_future»
- `server.hpp` — подключает оба варианта
- `main.cpp` — клиенты + выбор реализации аргументом
- `benchmark.cpp` — замеры

## Замеры: как выбрать лучший вариант

На **вашем сервере** запустите бенчмарк (чем больше `n_per_kind` и `repeats`, тем стабильнее среднее):

```bash
./build/task2_benchmark 300 7 task2_benchmark_results.csv
```

Откройте `task2_benchmark_results.csv`: сравните столбец `seconds_per_run_mean`. Меньше — лучше для данной нагрузки.

Затем запустите основную программу с **выигравшим** флагом, например:

```bash
./build/task2_client_server 500 promise-u
./build/task2_verify
```

Допустимые флаги реализации:

| Флаг        | Доставка результата | Контейнер по `id`   |
|-------------|---------------------|---------------------|
| `slot-u`    | mutex + cv          | `unordered_map`     |
| `slot-o`    | mutex + cv          | `std::map`          |
| `promise-u` | promise / shared_future | `unordered_map` |
| `promise-o` | promise / shared_future | `std::map`      |

По умолчанию в `task2_client_server` используется **`promise-u`**, если второй аргумент не указан (типичный учебный паттерн «очередь + promise»). Окончательный выбор за результатами `task2_benchmark` на вашей машине.

## Сборка и запуск

```bash
cd lab3/task2
cmake -S . -B build -G "Unix Makefiles"
cmake --build build -j
./build/task2_benchmark 200 5
./build/task2_client_server 100 promise-u
./build/task2_verify
```

## Запуск через тесты

```bash
cd build
ctest --output-on-failure
```

## Выходные файлы

- `client_sin.csv`, `client_sqrt.csv`, `client_pow.csv` — результаты клиентов
- `task2_benchmark_results.csv` — сводка замеров (после запуска `task2_benchmark`)
