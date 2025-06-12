#pragma once
#include "boost_pool.hpp"
#include "resources.hpp"
#include <vector>
#include <queue>
#include <stack>

namespace gmp { namespace containers {

    // type aliases 
    using atom_type_id_t = std::uint8_t;
    using atom_type_map_t = std::unordered_map<std::string, atom_type_id_t>;
    
    using namespace gmp::resources;

    template <typename T>
    class vec : public std::vector<T, resources::pool_allocator<T>> {
        using base = std::vector<T, resources::pool_allocator<T>>;        
    public:
        // Default constructor uses the default pool
        vec() : base(resources::pool_allocator<T>(get_default_pool())) {}

        template <typename size_type>
        explicit vec(size_type size, const T& value) : base(size, value, resources::pool_allocator<T>(get_default_pool())) {}
        
        // Constructor with specific pool
        explicit vec(resources::PoolType& pool) : base(resources::pool_allocator<T>(pool)) {}
        
        // copy constructor
        vec(const vec& other) : base(other) {}

        // move constructor
        vec(vec&& other) noexcept : base(std::move(other)) {}

        // copy assignment
        vec& operator=(const vec& other) { base::operator=(other); return *this; }

        // Forward all other vector constructors
        using base::base;
        
        // Inherit all vector operations
        using base::operator=;
    };

    template <typename T>
    using gmp_unique_ptr = resources::pool_unique_ptr<T, resources::PoolType>;

    // Helper function to create gmp_unique_ptr with default pool
    template <typename T, typename... Args>
    gmp_unique_ptr<T> make_gmp_unique(Args&&... args) {
        return gmp_unique_ptr<T>(resources::make_pool_unique<T>(get_default_pool(), std::forward<Args>(args)...));
    }

    // Helper function to create gmp_unique_ptr with specific pool
    template <typename T, typename... Args>
    gmp_unique_ptr<T> make_gmp_unique_with_pool(resources::PoolType& pool, Args&&... args) {
        return gmp_unique_ptr<T>(resources::make_pool_unique<T>(pool, std::forward<Args>(args)...));
    }

    template <typename T>
    class deque : public std::deque<T, resources::pool_allocator<T>> {
        using base = std::deque<T, resources::pool_allocator<T>>;
    public:
        // Default constructor uses the default pool
        deque() : base(resources::pool_allocator<T>(get_default_pool())) {}
        
        // Constructor with specific pool
        explicit deque(resources::PoolType& pool) : base(resources::pool_allocator<T>(pool)) {}
        
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

    // Queue implementation using custom allocator
    template <typename T>
    using queue = std::queue<T, deque<T>>;

    // Stack implementation using custom allocator
    template <typename T>
    using stack = std::stack<T, deque<T>>;

    template <typename Key, typename Value>
    class unordered_map : public std::unordered_map<Key, Value, std::hash<Key>, std::equal_to<Key>, 
        resources::pool_allocator<std::pair<const Key, Value>>> {
        using base = std::unordered_map<Key, Value, std::hash<Key>, std::equal_to<Key>, 
            resources::pool_allocator<std::pair<const Key, Value>>>;
    public:
        // Default constructor uses the default pool
        unordered_map() : base(resources::pool_allocator<std::pair<const Key, Value>>(get_default_pool())) {}
        
        // Constructor with specific pool
        explicit unordered_map(resources::PoolType& pool) 
            : base(resources::pool_allocator<std::pair<const Key, Value>>(pool)) {}
        
        // copy constructor
        unordered_map(const unordered_map& other) : base(other) {}

        // move constructor
        unordered_map(unordered_map&& other) noexcept : base(std::move(other)) {}

        // copy assignment
        unordered_map& operator=(const unordered_map& other) { base::operator=(other); return *this; }

        // Forward all other unordered_map constructors
        using base::base;
        
        // Inherit all unordered_map operations
        using base::operator=;
    };
}}