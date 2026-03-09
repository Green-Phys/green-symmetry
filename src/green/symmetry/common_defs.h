#ifndef GREEN_SYMMETRY_COMMON_DEFS_H
#define GREEN_SYMMETRY_COMMON_DEFS_H

#include <green/h5pp/archive.h>
#include <green/ndarray/ndarray.h>
#include <green/ndarray/ndarray_math.h>
#include <green/params/params.h>

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

namespace green::symmetry {
  template <size_t D>
  using itensor = green::ndarray::ndarray<int, D>;
  template <size_t D>
  using dtensor = green::ndarray::ndarray<double, D>;
  template <size_t D>
  using ztensor = green::ndarray::ndarray<std::complex<double>, D>;

  template <typename T>
  using Op_t = std::function<T(const T& val)>;

  template <typename T>
  Op_t<T> conj_op = [](const T& val) { return std::conj(val); };

  template <typename T>
  Op_t<T> no_op = [](const T& val) { return val; };

  // Matrix types
  template <typename prec>
  using MatrixX   = Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXcf = Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXd  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  // Matrix-Map types
  template <typename prec>
  using MMatrixX   = Eigen::Map<Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXcd = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXcf = Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using MMatrixXd  = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  // Const Matrix-Map types
  template <typename prec>
  using CMMatrixX   = Eigen::Map<const Eigen::Matrix<prec, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXcd = Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXcf = Eigen::Map<const Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using CMMatrixXd  = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  namespace details {
    /**
     * Find index of k-point `k` in reciprocal grid `kmesh`
     *
     * @param k - point to be found
     * @param kmesh - reciprocal space grid
     * @return index of `k` in `kmesh` or throws an error
     */
    inline int find_pos(const dtensor<1>& k, const dtensor<2>& kmesh) {
      for (size_t i = 0; i < kmesh.shape()[0]; ++i) {
        bool found = true;
        for (size_t j = 0; j < k.shape()[0]; ++j) {
          found &= std::abs(k(j) - kmesh(i, j)) < 1e-12;
        }
        if (found) {
          return i;
        }
      }
      throw std::logic_error("K point (" + std::to_string(k(0)) + ", " + std::to_string(k(1)) + ", " + std::to_string(k(2)) +
                             ") has not been found in the mesh.");
    }

    /**
     * Wrap k-point into the first Brillouin zone
     *
     * @param k - current k-point
     * @return k-point from the first Brillouin zone that is equivivalent to a given k-point
     */
    inline dtensor<1> wrap(const dtensor<1>& k) {
      dtensor<1> kk = k.copy();
      for (size_t j = 0; j < kk.shape()[0]; ++j) {
        while ((kk(j) - 9.9999999999e-1) > 0.0) {
          kk(j) -= 1.0;
        }
        if (std::abs(kk(j)) < 1e-9) {
          kk(j) = 0.0;
        }
        while (kk(j) < 0) {
          kk(j) += 1.0;
        }
      }
      return kk;
    };
  }  // namespace details

  /**
   * Define names of parameters that are needed brillouin_zone_utils
   * @param p - parameters object
   */
  inline void define_parameters(green::params::params& p) { p.define<std::string>("input_file", "Input file"); }
}

#endif // GREEN_SYMMETRY_COMMON_DEFS_H