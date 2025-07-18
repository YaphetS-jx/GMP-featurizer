#pragma once
#include "resources.hpp"
#include <rmm/mr/pinned_host_memory_resource.hpp>

namespace gmp { namespace resources {

    // Custom allocator that wraps pinned_host_memory_resource
    template <typename T>
    class pinned_host_allocator {
    public:
        using value_type = T;
        pinned_host_allocator() noexcept 
            : _pool(gmp::resources::gmp_resource::instance().get_pinned_host_memory_pool()) 
        {}

        // Allow rebinding to other types
        template <typename U>
        pinned_host_allocator(const pinned_host_allocator<U>&) noexcept {}

        T* allocate(size_t n) {
            if (n == 0) return nullptr;
            return static_cast<T*>(_pool->allocate(n * sizeof(T)));
        }

        void deallocate(T* p, size_t n) noexcept {
            if (p != nullptr) {
                _pool->deallocate(p, n);
            }
        }

        // All instances of this allocator are interchangeable:
        bool operator==(const pinned_host_allocator&) const noexcept { return true; }
        bool operator!=(const pinned_host_allocator& a) const noexcept { return !(*this == a); }
    private:
        rmm::mr::pool_memory_resource<rmm::mr::pinned_host_memory_resource>* _pool;
    };
}}