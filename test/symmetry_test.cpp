

#include <green/params/params.h>
#include <green/symmetry/symmetry.h>

#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <random>

using namespace std::string_literals;
using MatrixXcd = green::symmetry::MatrixXcd;

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

/**
 * @brief compare two 3d arrays of dimension (nso, nso) related by time-reversal operator
 * 
 * @tparam T - precision
 * @param array : 
 * @param array_trs 
 * @return true if array and array_trs form a time-reversal pair
 * @return false otherwise
 */
template <typename T>
bool compare_trs_matrices(const green::ndarray::ndarray<T,2>& array,
                          const green::ndarray::ndarray<T,2>& array_trs) {
  size_t nso = array.shape()[0];
  size_t nao = nso / 2;
  green::ndarray::ndarray<T,2> diff(nso, nso);
  for (size_t i = 0; i < nao; ++i) {
    for (size_t j = 0; j < nao; ++j) {
        diff(i, j) = std::conj(array(nao + i, nao + j));
        diff(nao + i, nao + j) = std::conj(array(i, j));
        diff(i, nao + j) = -1.0 * std::conj(array(nao + i, j));
        diff(nao + i, j) = -1.0 * std::conj(array(i, nao + j));
    }
  }
  diff = diff - array_trs;
  return std::all_of(diff.begin(), diff.end(), [](T a) { return std::abs(a) < 1e-12; });
}

bool is_unitary(const MatrixXcd& U, double tol = 1e-12) {
  if (U.rows() != U.cols()) {
    return false;
  }
  MatrixXcd identity = MatrixXcd::Identity(U.rows(), U.cols());
  return (U * U.adjoint()).isApprox(identity, tol) && (U.adjoint() * U).isApprox(identity, tol);
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

TEST_CASE("Brillouin Zone Utils") {
  SECTION("K-grid metadata") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/test.h5"s;
    std::string args       = "test --input_file " + input_file;
    green::symmetry::define_parameters(p);
    p.parse(args);

    size_t                     nk_ref  = 0;
    size_t                     ink_ref = 0;
    size_t                     nao_ref = 0;
    size_t                     nso_ref = 0;
    green::symmetry::dtensor<2> mesh_ref;
    green::h5pp::archive       in_file(input_file, "r");
    in_file["symmetry/k/nk"] >> nk_ref;
    in_file["symmetry/k/ink"] >> ink_ref;
    in_file["symmetry/k/mesh_scaled"] >> mesh_ref;
    in_file["params/nao"] >> nao_ref;
    in_file["params/nso"] >> nso_ref;
    in_file.close();

    green::symmetry::fermi_kmesh ksym(p);
    REQUIRE(ksym.nk() == nk_ref);
    REQUIRE(ksym.ink() == ink_ref);
    REQUIRE(ksym.mesh().shape() == mesh_ref.shape());
    REQUIRE(ksym.nao() == nao_ref);
    REQUIRE(ksym.nso() == nso_ref);
    REQUIRE(std::equal(ksym.mesh().begin(), ksym.mesh().end(), mesh_ref.begin(),
                       [](double a, double b) { return std::abs(a - b) < 1e-12; }));
  }

  SECTION("K-point stars") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/test.h5"s;
    std::string args       = "test --input_file " + input_file;
    green::symmetry::define_parameters(p);
    p.parse(args);

    size_t n_stars_ref = 0;
    green::h5pp::archive in_file(input_file, "r");
    in_file["symmetry/k/n_stars"] >> n_stars_ref;
    std::vector<std::vector<long>> stars_ref(n_stars_ref);
    for (size_t i = 0; i < n_stars_ref; ++i) {
      green::symmetry::itensor<1> star_i;
      in_file["symmetry/k/stars/" + std::to_string(i)] >> star_i;
      stars_ref[i] = std::vector<long>(star_i.data(), star_i.data() + star_i.size());
    }
    in_file.close();

    green::symmetry::fermi_kmesh ksym(p);

    // Number of stars matches file and equals number of irreducible points
    REQUIRE(ksym.n_stars() == n_stars_ref);
    REQUIRE(ksym.n_stars() == ksym.ink());

    // Each star matches the reference data loaded from file
    for (size_t i = 0; i < ksym.n_stars(); ++i) {
      REQUIRE(ksym.star(i).size() == stars_ref[i].size());
      REQUIRE(std::equal(ksym.star(i).begin(), ksym.star(i).end(), stars_ref[i].begin()));
    }

    // Stars partition the full BZ: every k-point appears exactly once
    std::vector<int> count(ksym.nk(), 0);
    for (size_t i = 0; i < ksym.n_stars(); ++i) {
      for (long k : ksym.star(i)) {
        REQUIRE(k >= 0);
        REQUIRE(static_cast<size_t>(k) < ksym.nk());
        count[k]++;
      }
    }
    for (size_t k = 0; k < ksym.nk(); ++k) {
      REQUIRE(count[k] == 1);
    }

    // Every k-point in star i maps back to irreducible point i via reduced_point()
    for (size_t i = 0; i < ksym.n_stars(); ++i) {
      for (long k : ksym.star(i)) {
        REQUIRE(ksym.reduced_point(static_cast<size_t>(k)) == i);
      }
    }

    // The representative full-BZ point of each irreducible point belongs to its own star
    for (size_t i = 0; i < ksym.n_stars(); ++i) {
      long rep = static_cast<long>(ksym.full_point(i));
      const auto& s = ksym.star(i);
      REQUIRE(std::find(s.begin(), s.end(), rep) != s.end());
    }
  }

  SECTION("Q-grid metadata") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/test.h5"s;
    std::string args       = "test --input_file " + input_file;
    green::symmetry::define_parameters(p);
    p.parse(args);

    size_t                     nq_ref  = 0;
    size_t                     inq_ref = 0;
    size_t                     naux_ref = 0;
    green::symmetry::dtensor<2> mesh_ref;
    green::h5pp::archive       in_file(input_file, "r");
    in_file["symmetry/q/nq"] >> nq_ref;
    in_file["symmetry/q/inq"] >> inq_ref;
    in_file["symmetry/q/mesh_scaled"] >> mesh_ref;
    in_file["params/NQ"] >> naux_ref;
    in_file.close();

    green::symmetry::bose_kmesh qsym(p);
    REQUIRE(qsym.nk() == nq_ref);
    REQUIRE(qsym.ink() == inq_ref);
    REQUIRE(qsym.mesh().shape() == mesh_ref.shape());
    REQUIRE(qsym.naux() == naux_ref);
    REQUIRE(std::equal(qsym.mesh().begin(), qsym.mesh().end(), mesh_ref.begin(),
                       [](double a, double b) { return std::abs(a - b) < 1e-12; }));

    MatrixXcd U_j2c;
    MatrixXcd U_p0;
    REQUIRE_NOTHROW(qsym.q_sym_transform_j2c(U_j2c, 0));
    REQUIRE_NOTHROW(qsym.q_sym_transform_p0(U_p0, 0));
    REQUIRE(U_j2c.rows() == static_cast<long>(naux_ref));
    REQUIRE(U_j2c.cols() == static_cast<long>(naux_ref));
    REQUIRE(U_p0.rows() == static_cast<long>(naux_ref));
    REQUIRE(U_p0.cols() == static_cast<long>(naux_ref));
  }

  SECTION("Symmetry operators are unitary") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/test.h5"s;
    std::string args       = "test --input_file " + input_file;
    green::symmetry::define_parameters(p);
    p.parse(args);

    green::symmetry::fermi_kmesh ksym(p);
    green::symmetry::bose_kmesh qsym(p);

    MatrixXcd U_k(ksym.nso(), ksym.nso());
    for (size_t k = 0; k < ksym.nk(); ++k) {
      ksym.k_sym_transform_ao(U_k, k);
      INFO("k-index = " << k);
      REQUIRE(is_unitary(U_k));
    }

    MatrixXcd U_q_j2c(qsym.naux(), qsym.naux());
    for (size_t q = 0; q < qsym.nk(); ++q) {
      qsym.q_sym_transform_j2c(U_q_j2c, q);
      INFO("q-index (j2c) = " << q);
      REQUIRE(is_unitary(U_q_j2c));
    }

    MatrixXcd U_q_p0(qsym.naux(), qsym.naux());
    for (size_t q = 0; q < qsym.nk(); ++q) {
      qsym.q_sym_transform_p0(U_q_p0, q);
      INFO("q-index (p0) = " << q);
      REQUIRE(is_unitary(U_q_p0));
    }
  }

  SECTION("Distinct k and q meshes from new layout") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/test_kq.h5"s;
    std::string args       = "test --input_file " + input_file;
    green::symmetry::define_parameters(p);
    p.parse(args);

    green::symmetry::fermi_kmesh ksym(p);
    green::symmetry::bose_kmesh qsym(p);

    REQUIRE(ksym.nk() == qsym.nk());
    REQUIRE_FALSE(std::equal(ksym.mesh().begin(), ksym.mesh().end(), qsym.mesh().begin(),
                       [](double a, double b) { return std::abs(a - b) < 1e-12; }));
    REQUIRE(ksym.ink() != qsym.ink());
  }

  SECTION("Initialization") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/test.h5"s;
    std::string args       = "test --input_file " + input_file;
    green::symmetry::define_parameters(p);
    p.parse(args);
    green::symmetry::brillouin_zone_utils bz(p);
  }

  SECTION("Fourier Transform") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/test.h5"s;
    std::string args       = "test --input_file " + input_file;
    green::symmetry::define_parameters(p);
    p.parse(args);
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
    green::symmetry::define_parameters(p);
    p.parse(args);
    green::symmetry::brillouin_zone_utils bz(p);
    green::symmetry::dtensor<1>           k1   = bz.kmesh()(0);
    green::symmetry::dtensor<1>           k2   = bz.kmesh()(5);
    green::symmetry::dtensor<1>           k3   = bz.kmesh()(11);
    green::symmetry::dtensor<1>           k4   = bz.kmesh()(bz.momentum_conservation({0, 5, 11})[3]);
    green::symmetry::dtensor<1>           diff = green::symmetry::details::wrap(k1 + k3 - k2 - k4);
    REQUIRE(std::all_of(diff.begin(), diff.end(), [](double a) { return std::abs(a) < 1e-12; }));
  }

  SECTION("K-point Operations") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/test.h5"s;
    std::string args       = "test --input_file " + input_file;
    green::symmetry::define_parameters(p);
    p.parse(args);
    green::symmetry::brillouin_zone_utils bz(p);
    green::symmetry::dtensor<1>           k1  = bz.kmesh()(4);
    int                                   pos = green::symmetry::details::find_pos(k1, bz.kmesh());
    auto                                  k2  = k1 + 0.1111;
    auto                                  k3  = k1 + 2.0;
    k3                                        = green::symmetry::details::wrap(k3);
    REQUIRE(pos == 4);
    REQUIRE_THROWS(green::symmetry::details::find_pos(k2, bz.kmesh()));
    REQUIRE(std::equal(k1.begin(), k1.end(), k3.begin(), [](double a, double b) { return std::abs(a - b) < 1e-12; }));
    std::complex<double> x(1,3);
    REQUIRE(std::abs(bz.nkpw() - double(1./bz.nk()))<1e-12);
  }

  SECTION("Symmetry") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/test.h5"s;
    std::string args       = "test --input_file " + input_file;
    green::symmetry::define_parameters(p);
    p.parse(args);
    green::symmetry::brillouin_zone_utils bz(p);
    size_t                      nao = bz.nao();
    green::symmetry::ztensor<4>           X(1, bz.ink(), nao, nao);
    initialize_array(X);
    const auto&                 cX = X;
    green::symmetry::ztensor<3> Z  = bz.ibz_to_full(X(0));
    green::symmetry::ztensor<3> W  = bz.full_to_ibz(Z);
    auto                        cZ = bz.ibz_to_full(cX(0));
    auto                        cW = bz.full_to_ibz(cZ);
    green::symmetry::MatrixXcd                   k_sym_op(nao, nao);
    green::symmetry::MatrixXcd                   Xk(nao, nao);
    k_sym_op.resize(nao, nao);
    for (size_t k = 0; k < bz.nk(); ++k) {
      size_t ik = bz.k_symmetry().reduced_point(k);
      bz.k_symmetry().k_sym_transform_ao(k_sym_op, k);
      for (size_t i = 0; i < nao; ++i) {
        for (size_t j = 0; j < nao; ++j) {
          Xk(i, j) = X(0, ik, i, j);
        }
      }
      Xk = k_sym_op * Xk * k_sym_op.adjoint();
      if (bz.k_symmetry().tr_conj_list()[k] != 0) {
        Xk = Xk.conjugate();
      }
      REQUIRE(std::equal(Xk.data(), Xk.data() + Xk.size(), Z(k).data(),
                         [&](const std::complex<double>& a, const std::complex<double>& b) { return std::abs(a - b) < 1e-12; }));
      REQUIRE(std::equal(Xk.data(), Xk.data() + Xk.size(), cZ(k).data(),
                         [&](const std::complex<double>& a, const std::complex<double>& b) { return std::abs(a - b) < 1e-12; }));
    }
    // Verify that ibz_to_full and full_to_ibz are consistent with each other
    REQUIRE(std::equal(X(0).begin(), X(0).end(), W.begin(),
                       [&](const std::complex<double>& a, const std::complex<double>& b) { return std::abs(a - b) < 1e-12; }));
    REQUIRE(std::equal(X(0).begin(), X(0).end(), cW.begin(),
                       [&](const std::complex<double>& a, const std::complex<double>& b) { return std::abs(a - b) < 1e-12; }));
    {
      auto Y = bz.k_symmetry().value_AO(X(0), 0);
      REQUIRE(std::equal(X(0, 0).begin(), X(0, 0).end(), Y.data(),
                     [&](const std::complex<double>& a, const std::complex<double>& b) { return std::abs(a - b) < 1e-12; }));
    }
    {
      auto Y = bz.k_symmetry().value_AO(X(0), 2);
      REQUIRE(std::equal(Z(2).begin(), Z(2).end(), Y.data(),
                     [&](const std::complex<double>& a, const std::complex<double>& b) { return std::abs(a - b) < 1e-12; }));
    }
  }

  /**
   * @brief tests the symmetry operations for time-reversal operation in X2C calculations where spin-orbit coupling is present
   */
  SECTION("Symmetry-X2C") {
    auto        p          = green::params::params("DESCR");
    std::string input_file = TEST_PATH + "/test_x2c.h5"s;
    std::string args       = "test --input_file " + input_file;
    green::symmetry::define_parameters(p);
    p.parse(args);
    green::symmetry::brillouin_zone_utils bz(p);
    size_t nao = bz.nao();
    size_t nso = bz.nso();
    REQUIRE(std::abs((double) nao - 18.0) <= 1e-12);
    REQUIRE(std::abs((double) nso - 36.0) <= 1e-12);
    green::symmetry::ztensor<4>           X(1, bz.ink(), nso, nso);
    initialize_array(X);
    const auto&                 cX = X;
    green::symmetry::ztensor<3> Z  = bz.ibz_to_full(X(0));
    green::symmetry::ztensor<3> W  = bz.full_to_ibz(Z);
    auto                        cZ = bz.ibz_to_full(cX(0));
    auto                        cW = bz.full_to_ibz(cZ);
    // For every TR-conjugate k-point, read the irreducible index from the mapping
    // and verify the X2C spin-flip relation against both Z and cZ.
    for (size_t k = 0; k < bz.nk(); ++k) {
      if (bz.k_symmetry().tr_conj_list()[k] != 0) {
        size_t ik = bz.k_symmetry().reduced_point(k);
        REQUIRE(compare_trs_matrices(X(0, ik), Z(k)));
        REQUIRE(compare_trs_matrices(X(0, ik), cZ(k)));
      }
    }
    REQUIRE(std::equal(X(0).begin(), X(0).end(), W.begin(),
                       [&](const std::complex<double>& a, const std::complex<double>& b) { return std::abs(a - b) < 1e-12; }));
    REQUIRE(std::equal(X(0).begin(), X(0).end(), cW.begin(),
                       [&](const std::complex<double>& a, const std::complex<double>& b) { return std::abs(a - b) < 1e-12; }));
  }
}