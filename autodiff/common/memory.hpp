//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2020 Allan Leal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#ifndef MEMORY_HPP_NX2YDBN0
#define MEMORY_HPP_NX2YDBN0

#include <cuda/std/type_traits>

namespace autodiff {
namespace detail {


template <typename T>
class shared_ptr {
    template <typename U>
    friend class shared_ptr;

    private:
        T* ptr;
        unsigned int* cnt;

        __host__ __device__ void initialize(bool increase_refcount) {
            if (cnt == nullptr) {
                cnt = new unsigned int(1);
            }
            else if (increase_refcount) {
                this->incref();
            }
        }

        __host__ __device__ void finalize() {
            if (this->decref()) {
                if (ptr != nullptr) {
                    delete ptr;
                    ptr = nullptr;
                }
                if (cnt != nullptr) {
                    delete cnt;
                    cnt = nullptr;
                }
            }
        }
        
        __host__ __device__ void incref() {
            ++(*cnt);
        }

        __host__ __device__ bool decref() {
            return --(*cnt) <= 0;
        }

    public:
        using element_type = cuda::std::remove_extent_t<T>;
        
        __host__ __device__ explicit shared_ptr() : ptr(nullptr), cnt(nullptr) { this->initialize(false); };

        __host__ __device__ explicit shared_ptr(T* p) : ptr(p), cnt(nullptr) { this->initialize(true); };
        
        __host__ __device__ shared_ptr(const shared_ptr &other) : ptr(other.ptr), cnt(other.cnt) {
            this->initialize(true);
        }

        template <typename U>
        __host__ __device__ shared_ptr(const shared_ptr<U> &other) : ptr(other.ptr), cnt(other.cnt) {
            this->initialize(true);
        }
        
        __host__ __device__ shared_ptr(const shared_ptr &other, T* p) : ptr(p), cnt(other.cnt) {
            this->initialize(true);
        }

        __host__ __device__ shared_ptr(shared_ptr&& other) : ptr(other.ptr), cnt(other.cnt) {
            other.ptr = nullptr;
            other.cnt = nullptr;
            this->initialize(false);
        }
        
        __host__ __device__ shared_ptr(shared_ptr &&other, T* p) = delete;
        
        __host__ __device__ ~shared_ptr() {
            this->finalize();
        }

        template <typename U>
        __host__ __device__ shared_ptr& operator=(const shared_ptr<U>& other) {
            if (&other == this) { return *this; }
            this->ptr = other.ptr;
            this->cnt = other.cnt;
            this->initialize(true);
            return *this;
        }
        
        template <typename U>
        __host__ __device__ shared_ptr& operator=(shared_ptr<U>&& other) {
            this->ptr = other.ptr;
            this->cnt = other.cnt;
            other.ptr = nullptr;
            other.cnt = nullptr;
            this->initialize(false);
            return *this;
        }
        
        __host__ __device__ T* get() const { return ptr; }
        __host__ __device__ T& operator*()  const { return *ptr; }
        __host__ __device__ T* operator->() const { return ptr; }
        __host__ __device__ operator bool() const { return ptr != nullptr; }
};


template<class T0, class T1>
__host__ __device__ bool operator==(shared_ptr<T0> const & lhs, shared_ptr<T1> const & rhs) {
    return lhs.get() == rhs.get();
}

template<class T>
__host__ __device__ bool operator==(shared_ptr<T> const & lhs, nullptr_t) {
    return lhs.get() == nullptr;
}

template<class T>
__host__ __device__ bool operator==(nullptr_t, shared_ptr<T> const & rhs) {
    return rhs.get() == nullptr;
}


template<class T0, class T1>
__host__ __device__ bool operator!=(shared_ptr<T0> const & lhs, shared_ptr<T1> const & rhs) {
    return lhs.get() != rhs.get();
}

template<class T>
__host__ __device__ bool operator!=(shared_ptr<T> const & lhs, nullptr_t) {
    return lhs.get() != nullptr;
}

template<class T>
__host__ __device__ bool operator!=(nullptr_t, shared_ptr<T> const & rhs) {
    return rhs.get() != nullptr;
}

template<class T, typename... Args>
__host__ __device__ shared_ptr<T> make_shared(Args&&... args) {
    return shared_ptr<T>(new T(args...));
}

template<class T, class U>
__host__ __device__ shared_ptr<T> static_pointer_cast(const shared_ptr<U>& r) {
    auto p = static_cast<typename shared_ptr<T>::element_type*>(r.get());
    return shared_ptr<T>(r, p);
}

template< class T, class U > 
__host__ __device__ shared_ptr<T> dynamic_pointer_cast(const shared_ptr<U>& r) noexcept
{
    if (auto p = dynamic_cast<typename shared_ptr<T>::element_type*>(r.get())) {
        return shared_ptr<T>(r, p);
    } else {
        return shared_ptr<T>();
    }
}

template< class T, class U >
__host__ __device__ shared_ptr<T> const_pointer_cast(const shared_ptr<U>& r) noexcept
{
    auto p = const_cast<typename shared_ptr<T>::element_type*>(r.get());
    return shared_ptr<T>(r, p);
}

template< class T, class U >
__host__ __device__ shared_ptr<T> reinterpret_pointer_cast(const shared_ptr<U>& r) noexcept
{
    auto p = reinterpret_cast<typename shared_ptr<T>::element_type*>(r.get());
    return shared_ptr<T>(r, p);
}

} /* end of namespace detail */
} /* end of namespace autodiff */

#endif /* end of include guard: MEMORY_HPP_NX2YDBN0 */

