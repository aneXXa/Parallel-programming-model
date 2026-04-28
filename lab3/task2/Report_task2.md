# Task 2 — отчёт

Полный текст раздела **Task 2** входит в общий отчёт лабораторной работы №3:

**[lab3/Report.md](../Report.md)** — раздел «Task 2».

Краткая справка по флагам реализации (дублирует таблицу в общем отчёте):

| Флаг | Доставка | Контейнер по `id` |
|------|----------|-------------------|
| `slot-u` | mutex + cv | `unordered_map` |
| `slot-o` | mutex + cv | `std::map` |
| `promise-u` | promise / shared_future | `unordered_map` |
| `promise-o` | promise / shared_future | `std::map` |
