#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>
#include <chrono>
#include <atomic>
#include "util.hpp"

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    mutable std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
    std::atomic<int> active_threads{0};
    
public:
    // Constructor creates the thread pool with specified number of threads
    ThreadPool(size_t threads) : stop(false) {
        threads = threads > 0 ? threads : gmp::util::get_system_thread_count();
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        
                        // Wait until there's a task or the pool is stopped
                        this->condition.wait(lock, [this] { 
                            return this->stop || !this->tasks.empty(); 
                        });
                        
                        // If stopped and no tasks, exit
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }
                        
                        // Get the next task
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    
                    // Execute the task
                    active_threads++;
                    task();
                    active_threads--;
                }
            });
        }
    }
    
    // Add a task to the pool
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            // Don't allow enqueueing after stopping the pool
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            
            tasks.emplace([task]() { (*task)(); });
        }
        
        condition.notify_one();
        return res;
    }
    
    // Get number of active threads
    int get_active_threads() const {
        return active_threads.load();
    }
    
    // Get total number of threads in the pool
    size_t get_thread_count() const {
        return workers.size();
    }
    
    // Get queue size
    size_t get_queue_size() const {
        std::unique_lock<std::mutex> lock(queue_mutex);
        return tasks.size();
    }
    
    // Destructor
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        
        condition.notify_all();
        
        for (std::thread &worker : workers) {
            worker.join();
        }
    }
};