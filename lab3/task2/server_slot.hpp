#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

namespace task2_detail {

template <typename T>
struct SlotResult {
    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;
    std::optional<T> value;
    std::exception_ptr error;
};

}  // namespace task2_detail

// Очередь + Map ожидающих результатов; доставка: mutex + condition_variable + optional.
template <typename T, typename PendingMap, typename Task = std::function<T()>>
class TaskServerSlot {
public:
    TaskServerSlot() = default;
    ~TaskServerSlot() { stop(); }

    void start() {
        std::lock_guard<std::mutex> lock(control_mutex_);
        if (running_) {
            return;
        }
        running_ = true;
        worker_ = std::thread(&TaskServerSlot::worker_loop, this);
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(control_mutex_);
            if (!running_) {
                return;
            }
            running_ = false;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    std::size_t add_task(Task task) {
        auto slot = std::make_shared<task2_detail::SlotResult<T>>();
        std::size_t id = 0;

        {
            std::lock_guard<std::mutex> lock(control_mutex_);
            if (!running_) {
                throw std::runtime_error("server is not running");
            }
            id = next_id_++;
            pending_.emplace(id, slot);
            tasks_.push(TaskItem{id, std::move(task)});
        }

        queue_cv_.notify_one();
        return id;
    }

    T request_result(std::size_t id) {
        std::shared_ptr<task2_detail::SlotResult<T>> slot;
        {
            std::lock_guard<std::mutex> lock(control_mutex_);
            auto it = pending_.find(id);
            if (it == pending_.end()) {
                throw std::runtime_error("unknown task id");
            }
            slot = it->second;
        }

        {
            std::unique_lock<std::mutex> lock(slot->mtx);
            slot->cv.wait(lock, [&slot]() { return slot->ready; });
            if (slot->error) {
                std::rethrow_exception(slot->error);
            }
        }

        T value;
        {
            std::lock_guard<std::mutex> lock(slot->mtx);
            value = *slot->value;
        }

        {
            std::lock_guard<std::mutex> lock(control_mutex_);
            pending_.erase(id);
        }
        return value;
    }

private:
    struct TaskItem {
        std::size_t id{};
        Task task;
    };

    void worker_loop() {
        while (true) {
            TaskItem item;
            std::shared_ptr<task2_detail::SlotResult<T>> slot;

            {
                std::unique_lock<std::mutex> lock(control_mutex_);
                queue_cv_.wait(lock, [this]() { return !tasks_.empty() || !running_; });
                if (tasks_.empty() && !running_) {
                    break;
                }
                item = std::move(tasks_.front());
                tasks_.pop();
                auto it = pending_.find(item.id);
                if (it == pending_.end()) {
                    continue;
                }
                slot = it->second;
            }

            try {
                T out = item.task();
                std::lock_guard<std::mutex> slot_lock(slot->mtx);
                slot->value = std::move(out);
                slot->ready = true;
            } catch (...) {
                std::lock_guard<std::mutex> slot_lock(slot->mtx);
                slot->error = std::current_exception();
                slot->ready = true;
            }
            slot->cv.notify_all();
        }
    }

    std::mutex control_mutex_;
    std::condition_variable queue_cv_;
    bool running_ = false;
    std::size_t next_id_ = 1;
    std::thread worker_;
    std::queue<TaskItem> tasks_;
    PendingMap pending_;
};

template <typename T, typename Task = std::function<T()>>
using TaskServerSlotUnordered = TaskServerSlot<
    T,
    std::unordered_map<std::size_t, std::shared_ptr<task2_detail::SlotResult<T>>>,
    Task>;

template <typename T, typename Task = std::function<T()>>
using TaskServerSlotOrdered = TaskServerSlot<
    T,
    std::map<std::size_t, std::shared_ptr<task2_detail::SlotResult<T>>>,
    Task>;
