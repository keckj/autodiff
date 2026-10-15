// Device tests (NVRTC) of the metaprogramming utilities and traits in autodiff/common.

#include <nvrtc_test.hpp>

#include <autodiff/common/binomialcoefficient.hpp>
#include <autodiff/common/classtraits.hpp>
#include <autodiff/common/meta.hpp>
#include <autodiff/common/numbertraits.hpp>
#include <autodiff/common/vectortraits.hpp>

using namespace autodiff;
using namespace autodiff::detail;

AUTODIFF_NVRTC_TEST(meta_loops)
{
    int s = 0;
    For<5>([&](auto i) { s += i; });
    CHECK(s == 10);

    int r = 0;
    ReverseFor<1, 4>([&](auto i) { r = 10 * r + int(i); });
    CHECK(r == 321);

    CHECK((Sum<0, 4>([](auto i) { return 2 * int(i); })) == 12);

    auto t = stdx::make_tuple(1, 2.5, 3.0f);
    double acc = 0;
    ForEach(t, [&](auto x) { acc += x; });
    CHECK(acc == 6.5);
    CHECK(Reduce(t, [](auto x) { return double(x) * 2; }) == 13.0);

    auto u = stdx::make_tuple(10, 20, 30);
    int dot = 0;
    ForEach(t, u, [&](auto a, auto b) { dot += int(a) * b; });
    CHECK(dot == 10 + 40 + 90);

    CHECK(TupleHead(t) == 1);
    auto tail = TupleTail(t);
    static_assert(TupleSize<decltype(tail)> == 2);
    CHECK(stdx::get<0>(tail) == 2.5);

    static_assert(Index<3>::index == 3);
    constexpr size_t three = Index<3>{};
    static_assert(three == 3);
}

AUTODIFF_NVRTC_TEST(meta_traits)
{
    static_assert(isArithmetic<double>);
    static_assert(!isArithmetic<stdx::array<double, 2>>);
    static_assert(isSame<NumericType<float>, float>);
    static_assert(Order<double> == 0);
    static_assert(isVector<stdx::array<double, 3>>);
    static_assert(isSame<VectorValueType<stdx::array<float, 3>>, float>);
    static_assert(isSame<VectorReplaceValueType<stdx::array<float, 3>, int>, stdx::array<int, 3>>);
    static_assert(hasSize<stdx::array<double, 3>>);
    static_assert(!hasSize<NumberTraits<double>>);
    static_assert(has_operator_bracket<stdx::array<double, 3>>::value);
    static_assert(isConvertible<int, double>);
    static_assert(isSame<CommonType<int, double>, double>);
    static_assert(BinomialCoefficient<5, 2> == 10.0);
    static_assert(BinomialCoefficient<50, 25> == 126410606437752.0);
    CHECK(true);
}
