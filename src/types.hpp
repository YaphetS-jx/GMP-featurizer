#pragma once
#include "boost_pool.hpp"
#include "resources.hpp"
#include <vector>

namespace gmp { namespace containers {

    // type aliases 
    using atom_type_id_t = std::uint8_t;
    using atom_type_map_t = std::unordered_map<std::string, atom_type_id_t>;
    
    using namespace gmp::resources;

    inline PoolType& get_default_pool() {
        return gmp_resource::instance().get_host_memory().get_pool();
    }

    template <typename T>
    class vec : public std::vector<T, resources::pool_allocator<T>> {
        using base = std::vector<T, resources::pool_allocator<T>>;
    public:
        // Default constructor uses the default pool
        vec() : base(resources::pool_allocator<T>(get_default_pool())) {}
        
        // Constructor with specific pool
        explicit vec(resources::PoolType& pool) : base(resources::pool_allocator<T>(pool)) {}
        
        // Forward all other vector constructors
        using base::base;
        
        // Inherit all vector operations
        using base::operator=;
    };

    template <typename T>
    class gmp_unique_ptr : public resources::pool_unique_ptr<T, resources::PoolType> {
        using base = resources::pool_unique_ptr<T, resources::PoolType>;
    public:
        // Default constructor creates null pointer with default pool
        gmp_unique_ptr() : base(nullptr, &get_default_pool()) {}
        
        // Constructor with specific pool
        explicit gmp_unique_ptr(resources::PoolType& pool) : base(nullptr, &pool) {}

        // Forward the main constructor
        gmp_unique_ptr(T* ptr, resources::PoolType* pool = &get_default_pool(), size_t n = 1) : base(ptr, pool, n) {}
        
        // Conversion constructor from base class
        gmp_unique_ptr(resources::pool_unique_ptr<T, resources::PoolType>&& other) 
            : base(other.release(), other.get_pool(), other.get_count()) {}
        
        // Move operations
        gmp_unique_ptr(gmp_unique_ptr&&) = default;
        gmp_unique_ptr& operator=(gmp_unique_ptr&&) = default;
        
        // Delete copy operations
        gmp_unique_ptr(const gmp_unique_ptr&) = delete;
        gmp_unique_ptr& operator=(const gmp_unique_ptr&) = delete;
    };

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
    
}}