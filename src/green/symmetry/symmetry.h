/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef GREEN_SYMMETRY_H
#define GREEN_SYMMETRY_H

#include "common_defs.h"
#include "except.h"
#include "symmetry_base.h"
#include "fermi_kmesh.h"
#include "bose_kmesh.h"

namespace green::symmetry {

  /**
   * @brief Encapsulates the q = k1 - k2 momentum conservation mapping
   *        between k-mesh and q-mesh.
   */
  class kq_map {
  public:
    kq_map(const symmetry_base& kpt_sym, const symmetry_base& qpt_sym);

    /** Full BZ k1-index in pair (k1, k2) with q = k1 - k2, i.e. k1 = k2 + q */
    size_t k1_from_k2q(size_t k2, size_t q) const { return _k1_from_k2q(k2, q); }

    /** Reduced BZ q-index for pair (k1, k2) */
    size_t q_from_k1k2(size_t k1, size_t k2) const { return _q_from_k1k2(k1, k2); }

    /** Full BZ k-index for pair (k1, k2) with k2 = k1 - q */
    size_t k2_from_k1q(size_t k1, size_t q) const { return _k2_from_k1q(k1, q); }

    // Full 4-momentum from a triplet via momentum conservation */

    /**
     * @brief returns full 4-momentum (k1, k2, k3, k4) for a given triplet (k1, k2, k3) using momentum conservation (k4 = k1 - k2 + k3)
     * 
     * @param n - input triplet of k-point indices (k1, k2, k3) in the full BZ
     * @return std::array<size_t, 4> 
     */
    std::array<size_t, 4> momentum_conservation(const std::array<size_t, 3>& n) const {
      size_t q = _q_from_k1k2(n[0], n[1]);
      size_t k4 = _k1_from_k2q(n[2], q);
      return {n[0], n[1], n[2], k4};
    }

  private:
    itensor<2> _k1_from_k2q;   // (nk x nk) -> k full BZ index
    itensor<2> _k2_from_k1q;   // (nk x nq) -> k full BZ index
    itensor<2> _q_from_k1k2;  // (nk x nk) -> q reduced BZ index
  };


  // No need to template this!
  class brillouin_zone_utils {
  public:
    brillouin_zone_utils(const green::params::params& p);

    /**
     * @return object describing system symmetries for k-mesh
     */
    const fermi_kmesh& k_symmetry() const { return _k_symmetry; }

    /**
     * @return object describing system symmetries for q-mesh
     */
    const bose_kmesh& q_symmetry() const { return _q_symmetry; }

    /**
     * @return object describing the k-q mapping
     */
    const kq_map& k_q_map() const { return _kq_map; }

    /**
     * @return number of k-points in the full first Brillouin zone
     */
    size_t nk() const { return _k_symmetry.nk(); }

    /**
     * @return number of q-points in the full q-mesh Brillouin zone
     */
    size_t nq() const { return _q_symmetry.nk(); }

    /**
     * @return number of k-points in the reduced first Brillouin zone
     */
    size_t ink() const { return _k_symmetry.ink(); }

    /**
     * @return number of q-points in the reduced q-mesh Brillouin zone
     */
    size_t inq() const { return _q_symmetry.ink(); }

    /**
     * @return number of AOs based on input file
     */
    size_t nao() const { return _k_symmetry.nao(); }

    /**
     * @return number of spin-orbitals based on input file
     */
    size_t nso() const { return _k_symmetry.nso(); }

    /**
     * @return number of auxiliary basis functions based on input file
     */
    size_t naux() const { return _q_symmetry.naux(); }
   
    /**
     * @return weight of each k-point (1/_nk)
     */
    double nkpw() const { return _nkpw; }

    /**
     * @return weight of each q-point (1/_nq)
     */
    double nqpw() const { return _nqpw; }

    /**
     * compute integral momenta using momentum conservation
     *
     * @param n - first integral momentum triplet
     * @return full 4 momenta for the current triplet
     */
    std::array<size_t, 4> momentum_conservation(const std::array<size_t, 3>& n) const {
      return _kq_map.momentum_conservation(n);
    }

    /**
     * @return reciprocal space grid
     */
    const dtensor<2>& kmesh() const { return _k_symmetry.mesh(); }

    const dtensor<2>& qmesh() const { return _q_symmetry.mesh(); }

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
     * Reconstruct array defined on the full first Brillouin zone from inpute array defined on reduced BZ.
     * First dimension should correspond to momentum index
     *
     * @tparam T - value type of array
     * @param val - array to be projected
     * @return new array on the full first BZ that corresponds to input array via defined symmetry relations
     */
    template <typename T>
    auto ibz_to_full(const green::ndarray::ndarray<T, 3>& val) const {
      using ST = std::remove_const_t<T>;
      assert(val.shape()[0] == _k_symmetry.ink());
      std::array<size_t, 3> new_shape(val.shape());
      new_shape[0] = _k_symmetry.nk();
      green::ndarray::ndarray<ST, 3> ret(new_shape);
      MatrixX<ST> k_sym_op;

      size_t nao = _k_symmetry.nao();
      size_t nk = _k_symmetry.nk();

      if (!_X2C){
        k_sym_op.resize(nao, nao);
        for (size_t k = 0; k < nk; ++k) {
          size_t ik = _k_symmetry.reduced_point(k);
          MMatrixX<ST> R_k(ret.data() + k * nao * nao, nao, nao);
          R_k = _k_symmetry.value_AO(val, k);
        }
      } else {
        // for X2C, spin-flip is necessary: X(-k) = Spin-flip [X(k).conj()]
        for (size_t k = 0; k < nk; ++k) {
          size_t ik = _k_symmetry.reduced_point(k);
          if (_k_symmetry.tr_conj_list()[k]) {
            for (size_t i = 0; i < nao; ++i) {
              for (size_t j = 0; j < nao; ++j) {
                  ret(k, i, j) = std::conj(val(ik, nao + i, nao + j));
                  ret(k, nao + i, nao + j) = std::conj(val(ik, i, j));
                  ret(k, i, nao + j) = -1.0 * std::conj(val(ik, nao + i, j));
                  ret(k, nao + i, j) = -1.0 * std::conj(val(ik, i, nao + j));
              }
            }
          } else {
              ret(k) << val(ik);
          }
        }
      }
      return ret;
    } // LCOV_EXCL_LINE

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
      assert(val.shape()[0] == _k_symmetry.nk());
      std::array<size_t, D> new_shape(val.shape());
      new_shape[0] = _k_symmetry.ink();
      green::ndarray::ndarray<std::remove_const_t<T>, D> ret(new_shape);
      for (size_t ik = 0; ik < _k_symmetry.ink(); ++ik) {
        size_t k = _k_symmetry.full_point(ik);
        ret(ik) << val(k);
      }
      return ret;
    } // LCOV_EXCL_LINE

  private:
    fermi_kmesh   _k_symmetry;
    bose_kmesh   _q_symmetry;
    kq_map      _kq_map;
    bool       _X2C;

    double     _nkpw;
    double     _nqpw;

    MatrixXcd  _T_k_to_r;
    MatrixXcd  _T_r_to_k;
  };
}  // namespace green::symmetry

#endif  // GREEN_SYMMETRY_H
