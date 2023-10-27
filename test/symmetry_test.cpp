

#include <green/params/params.h>
#include <green/symmetry/symmetry.h>

#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <random>

using namespace std::string_literals;

template <typename T, size_t Dim>
inline void initialize_array(green::ndarray::ndarray<T, Dim>& array) {
  // Specify the engine and distribution.
  std::mt19937                           mersenne_engine(1);  // Generates pseudo-random integers
  std::uniform_real_distribution<double> dist{0.0, 10.0};

  std::generate(array.begin(), array.end(), [&dist, &mersenne_engine]() -> T { return T(dist(mersenne_engine)); });
}

template <typename T, size_t Dim>
inline void initialize_array(green::ndarray::ndarray<std::complex<T>, Dim>& array) {
  // Specify the engine and distribution.
  std::mt19937                           mersenne_engine(1);  // Generates pseudo-random integers
  std::uniform_real_distribution<double> dist{0.0, 10.0};

  std::generate(array.begin(), array.end(), [&dist, &mersenne_engine]() -> std::complex<T> {
    return std::complex<T>(dist(mersenne_engine), dist(mersenne_engine));
  });
}

template <typename T, size_t D>
std::ostream& operator<<(std::ostream& os, const green::ndarray::ndarray<T, D>& array) {
  std::cout << "{";
  for (size_t i = 0; i < array.shape()[0]; ++i) {
    if constexpr (D == 1) {
      std::cout << array(i) << std::endl;
    } else {
      size_t rest_dim = std::accumulate(array.shape().begin() + 1, array.shape().end(), 1ul, std::multiplies<size_t>());
      for (size_t rest = 0; rest < rest_dim; ++rest) {
        std::cout << *(array.begin() + i * rest_dim + rest) << " ";
      }
      std::cout << std::endl;
    }
  }
  std::cout << "}" << std::endl;
  return os;  // returns the ostream
}

inline std::pair<int, char**> get_argc_argv(std::string& str) {
  std::string        key;
  std::vector<char*> splits = {(char*)str.c_str()};
  for (int i = 1; i < str.size(); i++) {
    if (str[i] == ' ') {
      str[i] = '\0';
      splits.emplace_back(&str[++i]);
    }
  }
  char** argv = new char*[splits.size()];
  for (int i = 0; i < splits.size(); i++) {
    argv[i] = splits[i];
  }

  return {(int)splits.size(), argv};
}

TEST_CASE("Brillouin Zone Utils") {
  SECTION("Initialization") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/test.h5"s;
    std::string args       = "test --input_file " + input_file;
    auto [argc, argv]      = get_argc_argv(args);
    p.parse(argc, argv);
    green::symmetry::brillouin_zone_utils bz(p);
  }

  SECTION("Fourier Transform") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/test.h5"s;
    std::string args       = "test --input_file " + input_file;
    auto [argc, argv]      = get_argc_argv(args);
    p.parse(argc, argv);
    green::symmetry::brillouin_zone_utils bz(p);
    green::symmetry::ztensor<4>           x_k(bz.nk(), 5, 5, 2);
    green::symmetry::ztensor<4>           x_r(bz.nk(), 5, 5, 2);
    green::symmetry::ztensor<4>           x_k_new(bz.nk(), 5, 5, 2);
    initialize_array(x_k);
    bz.k_to_r(x_k, x_r);
    bz.r_to_k(x_r, x_k_new);
    REQUIRE(std::equal(x_k.begin(), x_k.end(), x_k_new.begin(),
                       [](const std::complex<double>& a, const std::complex<double>& b) { return std::abs(a - b) < 1e-12; }));
  }

  SECTION("Momentum conservation") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/test.h5"s;
    std::string args       = "test --input_file " + input_file;
    auto [argc, argv]      = get_argc_argv(args);
    p.parse(argc, argv);
    green::symmetry::brillouin_zone_utils bz(p);
    green::symmetry::dtensor<1>           k1   = bz.mesh()(0);
    green::symmetry::dtensor<1>           k2   = bz.mesh()(5);
    green::symmetry::dtensor<1>           k3   = bz.mesh()(11);
    green::symmetry::dtensor<1>           k4   = bz.mesh()(bz.momentum_conservation({0, 5, 11})[3]);
    green::symmetry::dtensor<1>           diff = green::symmetry::details::wrap(k1 + k3 - k2 - k4);
    REQUIRE(std::all_of(diff.begin(), diff.end(), [](double a) { return std::abs(a) < 1e-12; }));
  }

  SECTION("Symmetry") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/test.h5"s;
    std::string args       = "test --input_file " + input_file;
    auto [argc, argv]      = get_argc_argv(args);
    p.parse(argc, argv);
    green::symmetry::brillouin_zone_utils bz(p);
    green::symmetry::ztensor<4>           X(bz.ink(), 5, 5, 2);
    initialize_array(X);
    green::symmetry::ztensor<4> Z = bz.ibz_to_full(X);
    green::symmetry::ztensor<4> W = bz.full_to_ibz(Z);
    REQUIRE(std::equal(X(1).begin(), X(1).end(), Z(2).begin(), [&](const std::complex<double>& a, const std::complex<double>& b) {
      return std::abs(a - std::conj(b)) < 1e-12;
    }));
    REQUIRE(std::equal(X.begin(), X.end(), W.begin(),
                       [&](const std::complex<double>& a, const std::complex<double>& b) { return std::abs(a - b) < 1e-12; }));
    {
      auto [Y, op] = bz.value(X, 0);
      auto op1     = op;
      REQUIRE(std::equal(X(0).begin(), X(0).end(), Y.begin(), [&](const std::complex<double>& a, const std::complex<double>& b) {
        return std::abs(a - op1(b)) < 1e-12;
      }));
    }
    {
      auto [Y, op] = bz.value(X, 2);
      auto op1     = op;
      REQUIRE(std::equal(Z(2).begin(), Z(2).end(), Y.begin(), [&](const std::complex<double>& a, const std::complex<double>& b) {
        return std::abs(a - op1(b)) < 1e-12;
      }));
    }
  }
}