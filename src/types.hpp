#pragma once
#include <vector>
#include <stack>
#include <unordered_map>
#include <string>
#include <cstdint>

namespace gmp { namespace containers {

    // type aliases 
    using atom_type_id_t = std::uint8_t;
    using atom_type_map_t = std::unordered_map<std::string, atom_type_id_t>;    

    template <typename T, typename Allocator = std::allocator<T>>
    class vector : public std::vector<T, Allocator> {
        using base = std::vector<T, Allocator>;
    public:
        // Default constructor uses the default allocator
        vector() : base() {}

        template <typename size_type>
        explicit vector(size_type size, const T& value) : base(size, value) {}

        template <typename size_type>
        explicit vector(size_type size, const T& value, const Allocator& alloc) : base(size, value, alloc) {}
        
        // Constructor with specific allocator
        explicit vector(const Allocator& alloc) : base(alloc) {}
        
        // copy constructor
        vector(const vector& other) : base(other) {}

        // move constructor
        vector(vector&& other) noexcept : base(std::move(other)) {}

        // copy assignment
        vector& operator=(const vector& other) { base::operator=(other); return *this; }

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