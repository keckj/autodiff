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

#pragma once

// pybind11 includes
#include <pybind11/pybind11.h>

// autodiff includes
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

#define PYBIND11_MAKE_OPAQUE_EIGEN_TYPES(scalar) \
    PYBIND11_MAKE_OPAQUE(autodiff::Vector##scalar##0th); \
    PYBIND11_MAKE_OPAQUE(autodiff::Vector##scalar##1st); \
    PYBIND11_MAKE_OPAQUE(autodiff::Vector##scalar##2nd); \
    PYBIND11_MAKE_OPAQUE(autodiff::Vector##scalar##3rd); \
    PYBIND11_MAKE_OPAQUE(autodiff::Vector##scalar##4th); \
    PYBIND11_MAKE_OPAQUE(autodiff::Matrix##scalar##0th); \
    PYBIND11_MAKE_OPAQUE(autodiff::Matrix##scalar##1st); \
    PYBIND11_MAKE_OPAQUE(autodiff::Matrix##scalar##2nd); \
    PYBIND11_MAKE_OPAQUE(autodiff::Matrix##scalar##3rd); \
    PYBIND11_MAKE_OPAQUE(autodiff::Matrix##scalar##4th); \
    PYBIND11_MAKE_OPAQUE(autodiff::Array##scalar##0th);  \
    PYBIND11_MAKE_OPAQUE(autodiff::Array##scalar##1st);  \
    PYBIND11_MAKE_OPAQUE(autodiff::Array##scalar##2nd);  \
    PYBIND11_MAKE_OPAQUE(autodiff::Array##scalar##3rd);  \
    PYBIND11_MAKE_OPAQUE(autodiff::Array##scalar##4th);  \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Vector##scalar##0th>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Vector##scalar##1st>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Vector##scalar##2nd>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Vector##scalar##3rd>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Vector##scalar##4th>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Matrix##scalar##0th>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Matrix##scalar##1st>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Matrix##scalar##2nd>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Matrix##scalar##3rd>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Matrix##scalar##4th>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Array##scalar##0th>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Array##scalar##1st>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Array##scalar##2nd>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Array##scalar##3rd>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<autodiff::Array##scalar##4th>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Vector##scalar##0th>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Vector##scalar##1st>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Vector##scalar##2nd>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Vector##scalar##3rd>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Vector##scalar##4th>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Matrix##scalar##0th>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Matrix##scalar##1st>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Matrix##scalar##2nd>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Matrix##scalar##3rd>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Matrix##scalar##4th>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Array##scalar##0th>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Array##scalar##1st>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Array##scalar##2nd>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Array##scalar##3rd>); \
    PYBIND11_MAKE_OPAQUE(Eigen::Ref<const autodiff::Array##scalar##4th>);

PYBIND11_MAKE_OPAQUE_EIGEN_TYPES(real);
PYBIND11_MAKE_OPAQUE_EIGEN_TYPES(dual);
