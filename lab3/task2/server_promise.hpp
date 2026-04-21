#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <future>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

namespace task2_detail {

template <typename T>
struct PromiseWait {
    std::promise<T> promise;
    std::shared_future<T> shared_result;

    PromiseWait() : shared_result(promise.get_future().share()) {}
};

}  // namespace task2_detail

// Очередь + Map ожидающих; доставка: std::promise / std::shared_future.
template <typename T, typename PendingMap, typename Task = std::function<T()>>
class TaskServerPromise {
public:
    TaskServerPromise() = default;
    ~TaskServerPromise() { stop(); }

    void start() {
        std::lock_guard<std::mutex> lock(control_mutex_);
        if (running_) {
            return;
        }
        running_ = true;
        worker_ = std::thread(&TaskServerPromise::worker_loop, this);
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
        std::size_t id = 0;
        auto slot = std::make_unique<task2_detail::PromiseWait<T>>();

        {
            std::lock_guard<std::mutex> lock(control_mutex_);
            if (!running_) {
                throw std::runtime_error("server is not running");
            }
            id = next_id_++;
            pending_.emplace(id, std::move(slot));
            tasks_.push(TaskItem{id, std::move(task)});
        }

        queue_cv_.notify_one();
        return id;
    }

    T request_result(std::size_t id) {
        std::shared_future<T> fut;
        {
            std::lock_guard<std::mutex> lock(control_mutex_);
            auto it = pending_.find(id);
            if (it == pending_.end()) {
                throw std::runtime_error("unknown task id");
            }
            fut = it->second->shared_result;
        }

        T value = fut.get();

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

            {
                std::unique_lock<std::mutex> lock(control_mutex_);
                queue_cv_.wait(lock, [this]() { return !tasks_.empty() || !running_; });
                if (tasks_.empty() && !running_) {
                    break;
                }
                item = std::move(tasks_.front());
                tasks_.pop();
                if (pending_.find(item.id) == pending_.end()) {
                    continue;
                }
            }

            try {
                T out = item.task();
                std::lock_guard<std::mutex> lock(control_mutex_);
                auto it = pending_.find(item.id);
                if (it != pending_.end()) {
                    it->second->promise.set_value(std::move(out));
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(control_mutex_);
                auto it = pending_.find(item.id);
                if (it != pending_.end()) {
                    it->second->promise.set_exception(std::current_exception());
                }
            }
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
using TaskServerPromiseUnordered = TaskServerPromise<
    T,
    std::unordered_map<std::size_t, std::unique_ptr<task2_detail::PromiseWait<T>>>,
    Task>;

template <typename T, typename Task = std::function<T()>>
using TaskServerPromiseOrdered = TaskServerPromise<
    T,
    std::map<std::size_t, std::unique_ptr<task2_detail::PromiseWait<T>>>,
    Task>;
