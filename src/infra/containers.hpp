#pragma once
#include <vector>
#include <stack>
#include <unordered_map>
#include <string>
#include <cstdint>
#ifdef GMP_ENABLE_CUDA
#include "resources.hpp"
#include <rmm/device_uvector.hpp>
#endif

namespace gmp { namespace containers {

#ifndef GMP_ENABLE_CUDA
    template <typename T>
    using vector = std::vector<T, std::allocator<T>>;
#else
    template <typename T>
    using vector = std::vector<T, gmp::resources::pinned_host_allocator<T>>;
    
    template <typename T>
    using vector_device = rmm::device_uvector<T>;
#endif

    template <typename T, typename Allocator = std::allocator<T>>
    class deque : public std::deque<T, Allocator> {
        using base = std::deque<T, Allocator>;
    public:
        // Default constructor uses the default pool
        deque() : base() {}
        
        // Constructor with specific pool
        explicit deque(const Allocator& alloc) : base(alloc) {}
        
        // copy constructor
        deque(const deque& other) : base(other) {}

        // move constructor
        deque(deque&& other) noexcept : base(std::move(other)) {}

        // copy assignment
        deque& operator=(const deque& other) { base::operator=(other); return *this; }

        // Forward all other deque constructors
        using base::base;
        
        // Inherit all deque operations
        using base::operator=;
    };

    // Stack implementation using custom allocator
    template <typename T, typename Allocator = std::allocator<T>>
    using stack = std::stack<T, deque<T, Allocator>>;
}}