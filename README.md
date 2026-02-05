# Parallel-programming-model

## Task 1 — Sinus

Массив из 10⁷ элементов заполняется значениями синуса (один период на всю длину). Вычисляется сумма элементов, результат выводится в терминал.

### Выбор типа массива при сборке

Тип элементов массива задаётся опцией CMake `USE_DOUBLE`:

| Тип массива | Команда конфигурации |
|-------------|----------------------|
| **double** (по умолчанию) | `cmake -B build` или `cmake -B build -DUSE_DOUBLE=ON` |
| **float** | `cmake -B build -DUSE_DOUBLE=OFF` |

### Сборка и запуск

Из корня репозитория:

```bash
# double (по умолчанию)
cmake -B build
cmake --build build
./build/task1/sinus

# float
cmake -B build -DUSE_DOUBLE=OFF
cmake --build build
./build/task1/sinus
```

Из папки `task1`:

```bash
cd task1
cmake -B build -DUSE_DOUBLE=ON   # double
cmake -B build -DUSE_DOUBLE=OFF  # float
cmake --build build
./build/sinus
```

### Результаты (вывод в терминал)

После сборки и запуска программа выводит тип массива, число элементов и сумму. Пример:

**double** (ожидаемо: сумма близка к 0, малая погрешность):
```
Array type: double
Elements:   10000000
Sum:        <запустите и подставьте значение>
```

**float** (ожидаемо: сумма близка к 0, погрешность обычно больше чем у double):
```
Array type: float
Elements:   10000000
Sum:        <запустите и подставьте значение>
```

Теоретически сумма одного периода синуса равна 0; из-за погрешности округления получаются малые ненулевые значения. После запуска обоих вариантов подставьте полученные `Sum` в README при желании.
