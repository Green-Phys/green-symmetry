/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef GF2_BZ_UTILS_H
#define GF2_BZ_UTILS_H

#include <green/h5pp/archive.h>
#include <green/ndarray/ndarray.h>
#include <green/ndarray/ndarray_math.h>
#include <green/params/params.h>

#include <Eigen/Dense>
#include <algorithm>

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

  class inv_symm_op {
  public:
    inv_symm_op(const green::params::params& p) {
      dtensor<2>           kmesh;
      std::vector<long>    irre_list;
      std::vector<long>    index;
      green::h5pp::archive in_file(p["input_file"], "r");
      in_file["grid/weight"] >> _weight;
      in_file["grid/index"] >> index;
      in_file["grid/ir_list"] >> _reduced_to_full;
      in_file["grid/conj_list"] >> _conj_list;
      in_file["grid/num_kpair_stored"] >> _num_kpair_stored;
      in_file["grid/conj_pairs_list"] >> _conj_kpair_list;
      in_file["grid/trans_pairs_list"] >> _trans_kpair_list;
      in_file["grid/kpair_irre_list"] >> _kpair_irre_list;
      in_file.close();
      _full_to_reduced.resize(index.size());
      long k_ir;
      for (size_t i = 0; i < index.size(); ++i) {
        k_ir = -1;
        for (size_t j = 0; j < _reduced_to_full.size(); ++j) {
          if (_reduced_to_full[j] == index[i]) {
            k_ir = j;
            break;
          }
        }
        if (k_ir < 0) {
          throw std::runtime_error("Can not find corresponding point in the reduced BZ");
        }
        _full_to_reduced[i] = k_ir;
      }
    }

    /**
     * @return vector that maps k-point from the full first BZ to a k-point in the reduced first BZ
     */
    const std::vector<size_t>& full_to_reduced() const { return _full_to_reduced; }
    /**
     * @return vector that maps k-point from the reduced first BZ to a k-point in the full first BZ
     */
    const std::vector<size_t>& reduced_to_full() const { return _reduced_to_full; }
    /**
     * @return list of flags for each k-point in the full first BZ telling whether we need to du complex conjugations
     * to get corresponding point in reduced BZ
     */
    const std::vector<long>&   conj_list() const { return _conj_list; }

    size_t                     num_kpair_stored() const { return _num_kpair_stored; }

    const std::vector<long>&   conj_kpair_list() const { return _conj_kpair_list; }

    const std::vector<long>&   trans_kpair_list() const { return _trans_kpair_list; }

    const std::vector<long>&   kpair_irre_list() const { return _kpair_irre_list; }

    /**
     * @return list of weight for each k-point in the full first BZ (w = 2 if k-point has exact mapping to reduced BZ, 0 otherwise)
     */
    const std::vector<double>& weight() const { return _weight; }

    /**
     * Find position in irre_list
     */
    size_t irre_pos(size_t k) const {
      auto   itr   = std::find(_reduced_to_full.begin(), _reduced_to_full.end(), k);
      size_t index = std::distance(_reduced_to_full.begin(), itr);
      return index;
    }

    size_t irre_pos_kpair(size_t idx) const {
      auto   itr   = std::find(_kpair_irre_list.begin(), _kpair_irre_list.end(), idx);
      size_t index = std::distance(_kpair_irre_list.begin(), itr);
      return index;
    }

    /**
     * For a given value from reduced first BZ and k-point from the full first BZ,
     * determine the corresponding value in the full first BZ
     *
     * @param val - value from reduced  first BZ
     * @param k - k-point to compute actual value for
     * @return corresponding value in the full first BZ
     */
    std::complex<double> value(std::complex<double> val, size_t k) const { return _conj_list[k] == 0 ? val : std::conj(val); }

    template <typename T>
    Op_t<T> op(size_t k) const {
      return _conj_list[k] ? conj_op<T> : no_op<T>;
    }

    /**
     * For a given index `k` of a point from the full first BZ return an index `ik` of a corresponding point
     * from the reduced first BZ
     *
     * @param k - index of a point in the full BZ
     * @return index of a corresponding point in the reduced BZ
     */
    size_t reduced_point(size_t k) const { return _full_to_reduced[k]; }

    /**
     * For a given index `ik` of a point from the reduced first BZ return an index `k` of a corresponding point
     * from the reduced first BZ
     *
     * @param ik - index of a point in the reduced BZ
     * @return index of a corresponding point in the full BZ
     */
    size_t full_point(size_t ik) const { return _reduced_to_full[ik]; }

  private:
    // Mapping of k-point from full BZ to reduced BZ
    std::vector<size_t> _full_to_reduced;
    // Mapping of k-point from reduced BZ to full BZ
    std::vector<size_t> _reduced_to_full;
    // weight of the k-point in reduced BZ
    std::vector<double> _weight;
    // conjugate list
    std::vector<long> _conj_list;

    // k-pairs information
    std::vector<long> _conj_kpair_list;
    std::vector<long> _trans_kpair_list;
    std::vector<long> _kpair_irre_list;
    size_t            _num_kpair_stored;
  };

  template <typename Symmetry = inv_symm_op>
  class brillouin_zone_utils {
  public:
    brillouin_zone_utils(const green::params::params& p);

    /**
     * @return object describing system symmetries
     */
    const Symmetry& symmetry() const {return _symmetry;}
    /**
     * @return number of k-points in the full first Brillouin zone
     */
    size_t nk() const { return _nk; }

    /**
     * @return number of k-points in the reduced first Brillouin zone
     */
    size_t ink() const { return _ink; }

    /**
     * @return weight of each k-point (1/_nk)
     */
    double nkpw() const { return _nkpw; }

    /**
     * compute integral momenta using momentum conservation
     *
     * @param n - first integral momentum triplet
     * @return full 4 momenta for the current triplet
     */
    std::array<size_t, 4> momentum_conservation(const std::array<size_t, 3>& n) const;

    /**
     * @return reciprocal space grid
     */
    const dtensor<2>& mesh() const { return _kmesh; }

    /**
     * Using discrete Fourier transform tensor `in` into a real space tensor `out`
     *
     * @tparam tensor_t - tensor type
     * @param in - input tensor in momentum space
     * @param out - output tensor in real space
     */
    template <typename tensor_t>
    void k_to_r(const tensor_t& in, tensor_t& out) const;

    /**
     * Using discrete Fourier transform tensor `in` into a momentum space tensor `out`
     *
     * @tparam tensor_t - tensor type
     * @param in - input tensor in real space
     * @param out - output tensor in momentum space
     */
    template <typename tensor_t>
    void r_to_k(const tensor_t& in, tensor_t& out) const;

    /**
     * Using defined symmetry operation transform the `value` for a given k-point
     *
     * @param val
     * @param k
     * @return
     */
    std::complex<double> value(std::complex<double> val, size_t k) const { return _symmetry.value(val, k); }

    /**
     * Using defined symmetry operation transform the `value` for a given k-point
     *
     * @param val
     * @param k
     * @return
     */
    template <typename T, size_t D>
    auto value(const green::ndarray::ndarray<T, D>& val, size_t k) const {
      assert(val.shape()[0] == _ink);
      return std::make_pair(val(_symmetry.reduced_point(k)), _symmetry.template op<T>(k));
    }

    /**
     * Reconstruct array defined on the full first Brillouin zone from inpute array defined on reduced BZ.
     * First dimension should correspond to momentum index
     *
     * @tparam T - value type of array
     * @tparam D - dimension of array
     * @param val - array to be projected
     * @return new array on the full first BZ that corresponds to input array via defined symmetry relations
     */
    template <typename T, size_t D>
    auto ibz_to_full(const green::ndarray::ndarray<T, D>& val) const {
      assert(val.shape()[0] == _ink);
      std::array<size_t, D> new_shape(val.shape());
      new_shape[0] = _nk;
      green::ndarray::ndarray<T, D> ret(new_shape);
      for (size_t k = 0; k < _nk; ++k) {
        size_t ik = _symmetry.reduced_point(k);
        std::transform(val(ik).begin(), val(ik).end(), ret(k).begin(),
                       [this, k](const T& item) { return _symmetry.template op<T>(k)(item); });
      }
      return ret;
    }

    /**
     * Project input array defined on the full first Brillouin zone onto reduced one. First dimension should correspond to
     * momentum index
     *
     * @tparam T - value type of array
     * @tparam D - dimension of array
     * @param val - array to be projected
     * @return new array that corresponds to input array via defined symmetry relations
     */
    template <typename T, size_t D>
    auto full_to_ibz(const green::ndarray::ndarray<T, D>& val) const {
      assert(val.shape()[0] == _nk);
      std::array<size_t, D> new_shape(val.shape());
      new_shape[0] = _ink;
      green::ndarray::ndarray<T, D> ret(new_shape);
      for (size_t ik = 0; ik < _ink; ++ik) {
        size_t k = _symmetry.full_point(ik);
        ret(ik) << val(k);
      }
      return ret;
    }

  private:
    Symmetry   _symmetry;
    size_t     _nk;
    size_t     _ink;

    double     _nkpw;

    itensor<2> _q_ind;
    itensor<2> _q_ind2;

    dtensor<2> _kmesh;

    MatrixXcd  _T_k_to_r;
    MatrixXcd  _T_r_to_k;

    size_t     mom_cons(size_t i, size_t j, size_t k) const;
  };
}  // namespace green::symmetry

#endif  // GF2_BZ_UTILS_H
