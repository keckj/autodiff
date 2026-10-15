//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright © 2018–2024 Allan Leal
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

#pragma once

// autodiff includes
#include <autodiff/common/config.hpp>
#include <autodiff/common/stdlib.hpp>
#include <autodiff/common/meta.hpp>

//======================================================================================================================
// Minimal shared_ptr and function types usable in host and device code (std::shared_ptr and std::function are not
// available in device code). They are used by the reverse mode (autodiff::var) when compiling with a CUDA compiler.
//
// Unlike their std counterparts:
// - the reference count is not atomic: an expression tree must not be shared among threads (as with std::shared_ptr,
//   autodiff::var is not thread safe anyway since computing derivatives mutates the expression tree);
// - objects created in device code (using the device heap, see cudaLimitMallocHeapSize) can only be used and destroyed
//   in device code, and objects created in host code only in host code;
// - function<R(Args...)> shares its callable among copies instead of copying it.
//======================================================================================================================

namespace autodiff {
namespace detail {

/// The control block of a shared_ptr: the reference count and how to destroy the owned object.
struct SharedCount
{
    long count;
    void* owned;
    void (*destroy)(void*);
};

template<typename U>
AUTODIFF_DEVICE_FUNC void sharedDestroy(void* p) { delete static_cast<U*>(p); }

template<typename T>
class shared_ptr
{
    template<typename U> friend class shared_ptr;

    T* ptr = nullptr;
    SharedCount* cnt = nullptr;

    AUTODIFF_DEVICE_FUNC void incref() const { if(cnt) ++cnt->count; }

    AUTODIFF_DEVICE_FUNC void release()
    {
        if(cnt && --cnt->count == 0)
        {
            cnt->destroy(cnt->owned);
            delete cnt;
        }
        ptr = nullptr;
        cnt = nullptr;
    }

public:
    using element_type = stdx::remove_extent_t<T>;

    AUTODIFF_DEVICE_FUNC constexpr shared_ptr() noexcept {}

    AUTODIFF_DEVICE_FUNC constexpr shared_ptr(decltype(nullptr)) noexcept {}

    /// Take ownership of p (allocated with new).
    template<typename U, Requires<stdx::is_convertible_v<U*, T*>> = true>
    AUTODIFF_DEVICE_FUNC explicit shared_ptr(U* p) : ptr(p)
    {
        if(p) cnt = new SharedCount{1, static_cast<void*>(p), &sharedDestroy<U>};
    }

    /// Aliasing constructor: share the ownership of other but point to p.
    template<typename U>
    AUTODIFF_DEVICE_FUNC shared_ptr(const shared_ptr<U>& other, T* p) noexcept : ptr(p), cnt(other.cnt) { incref(); }

    AUTODIFF_DEVICE_FUNC shared_ptr(const shared_ptr& other) noexcept : ptr(other.ptr), cnt(other.cnt) { incref(); }

    AUTODIFF_DEVICE_FUNC shared_ptr(shared_ptr&& other) noexcept : ptr(other.ptr), cnt(other.cnt)
    {
        other.ptr = nullptr;
        other.cnt = nullptr;
    }

    template<typename U, Requires<stdx::is_convertible_v<U*, T*>> = true>
    AUTODIFF_DEVICE_FUNC shared_ptr(const shared_ptr<U>& other) noexcept : ptr(other.ptr), cnt(other.cnt) { incref(); }

    template<typename U, Requires<stdx::is_convertible_v<U*, T*>> = true>
    AUTODIFF_DEVICE_FUNC shared_ptr(shared_ptr<U>&& other) noexcept : ptr(other.ptr), cnt(other.cnt)
    {
        other.ptr = nullptr;
        other.cnt = nullptr;
    }

    AUTODIFF_DEVICE_FUNC ~shared_ptr() { release(); }

    AUTODIFF_DEVICE_FUNC shared_ptr& operator=(shared_ptr other) noexcept
    {
        swap(other);
        return *this;
    }

    AUTODIFF_DEVICE_FUNC void swap(shared_ptr& other) noexcept
    {
        T* p = ptr; ptr = other.ptr; other.ptr = p;
        SharedCount* c = cnt; cnt = other.cnt; other.cnt = c;
    }

    AUTODIFF_DEVICE_FUNC void reset() noexcept { release(); }

    AUTODIFF_DEVICE_FUNC T* get() const noexcept { return ptr; }
    AUTODIFF_DEVICE_FUNC T& operator*() const noexcept { return *ptr; }
    AUTODIFF_DEVICE_FUNC T* operator->() const noexcept { return ptr; }
    AUTODIFF_DEVICE_FUNC long use_count() const noexcept { return cnt ? cnt->count : 0; }
    AUTODIFF_DEVICE_FUNC explicit operator bool() const noexcept { return ptr != nullptr; }
};

template<typename T, typename U> AUTODIFF_DEVICE_FUNC bool operator==(const shared_ptr<T>& l, const shared_ptr<U>& r) { return l.get() == r.get(); }
template<typename T, typename U> AUTODIFF_DEVICE_FUNC bool operator!=(const shared_ptr<T>& l, const shared_ptr<U>& r) { return l.get() != r.get(); }
template<typename T> AUTODIFF_DEVICE_FUNC bool operator==(const shared_ptr<T>& l, decltype(nullptr)) { return l.get() == nullptr; }
template<typename T> AUTODIFF_DEVICE_FUNC bool operator==(decltype(nullptr), const shared_ptr<T>& r) { return r.get() == nullptr; }
template<typename T> AUTODIFF_DEVICE_FUNC bool operator!=(const shared_ptr<T>& l, decltype(nullptr)) { return l.get() != nullptr; }
template<typename T> AUTODIFF_DEVICE_FUNC bool operator!=(decltype(nullptr), const shared_ptr<T>& r) { return r.get() != nullptr; }

template<typename T, typename... Args>
AUTODIFF_DEVICE_FUNC shared_ptr<T> make_shared(Args&&... args)
{
    return shared_ptr<T>(new T(stdx::forward<Args>(args)...));
}

template<typename T, typename U>
AUTODIFF_DEVICE_FUNC shared_ptr<T> static_pointer_cast(const shared_ptr<U>& r) noexcept
{
    return shared_ptr<T>(r, static_cast<T*>(r.get()));
}

template<typename T, typename U>
AUTODIFF_DEVICE_FUNC shared_ptr<T> const_pointer_cast(const shared_ptr<U>& r) noexcept
{
    return shared_ptr<T>(r, const_cast<T*>(r.get()));
}

template<typename T, typename U>
AUTODIFF_DEVICE_FUNC shared_ptr<T> reinterpret_pointer_cast(const shared_ptr<U>& r) noexcept
{
    return shared_ptr<T>(r, reinterpret_cast<T*>(r.get()));
}

template<typename Signature>
class function;

/// A copyable type erased callable (the callable is shared among copies).
template<typename R, typename... Args>
class function<R(Args...)>
{
    struct Callable
    {
        AUTODIFF_DEVICE_FUNC virtual ~Callable() {}
        AUTODIFF_DEVICE_FUNC virtual R call(Args... args) = 0;
    };

    template<typename F>
    struct CallableImpl : Callable
    {
        F f;
        AUTODIFF_DEVICE_FUNC CallableImpl(F&& fn) : f(stdx::move(fn)) {}
        AUTODIFF_DEVICE_FUNC CallableImpl(const F& fn) : f(fn) {}
        AUTODIFF_DEVICE_FUNC R call(Args... args) override { return f(stdx::forward<Args>(args)...); }
    };

    shared_ptr<Callable> callable;

public:
    AUTODIFF_DEVICE_FUNC function() {}

    template<typename F, Requires<!stdx::is_same_v<stdx::decay_t<F>, function>> = true>
    AUTODIFF_DEVICE_FUNC function(F&& f) : callable(make_shared<CallableImpl<stdx::decay_t<F>>>(stdx::forward<F>(f))) {}

    AUTODIFF_DEVICE_FUNC R operator()(Args... args) const { return callable->call(stdx::forward<Args>(args)...); }

    AUTODIFF_DEVICE_FUNC explicit operator bool() const { return static_cast<bool>(callable); }
};

} // namespace detail
} // namespace autodiff
