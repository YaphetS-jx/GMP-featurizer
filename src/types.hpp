#pragma once
#include "boost_pool.hpp"
#include "resources.hpp"
#include <vector>
#include <stack>

namespace gmp { namespace containers {

    // type aliases 
    using atom_type_id_t = std::uint8_t;
    using atom_type_map_t = std::unordered_map<std::string, atom_type_id_t>;
    
    using namespace gmp::resources;

    template <typename T, typename Allocator = std::allocator<T>>
    class vec : public std::vector<T, Allocator> {
        using base = std::vector<T, Allocator>;
    public:
        // Default constructor uses the default allocator
        vec() : base() {}

        template <typename size_type>
        explicit vec(size_type size, const T& value) : base(size, value) {}

        template <typename size_type>
        explicit vec(size_type size, const T& value, const Allocator& alloc) : base(size, value, alloc) {}
        
        // Constructor with specific allocator
        explicit vec(const Allocator& alloc) : base(alloc) {}
        
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