/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include <green/params/params.h>

#include <complex>

#include "green/symmetry/symmetry.h"

using namespace std::literals::complex_literals;

namespace green::symmetry {

  template <typename Symmetry>
  brillouin_zone_utils<Symmetry>::brillouin_zone_utils(const green::params::params& p) : _symmetry(p) {
    dtensor<2>           kmesh;
    green::h5pp::archive in_file(p["input_file"], "r");
    in_file["grid/nk"] >> _nk;
    in_file["grid/ink"] >> _ink;
    in_file["grid/k_mesh_scaled"] >> kmesh;
    in_file["params/nao"] >> _nao;
    in_file["params/nso"] >> _nso;
    in_file.close();

    _X2C = false;
    if (_nao != _nso) {_X2C = true;}

    _T_k_to_r.resize(_nk, _nk);
    _T_r_to_k.resize(_nk, _nk);
    _q_ind2.resize(_nk, _nk);
    _q_ind.resize(_nk, _nk);
    _nkpw   = 1.0 / double(_nk);

    int nkx = std::cbrt(double(_nk));
    assert(nkx * nkx * nkx == _nk);
    dtensor<2> rmesh_scaled(_nk, 3);
    for (int r = 0; r < _nk; ++r) {
      int rx             = r / (nkx * nkx);
      int ry             = (r / nkx) % nkx;
      int rz             = r % nkx;
      rmesh_scaled(r, 0) = double(rx);
      rmesh_scaled(r, 1) = double(ry);
      rmesh_scaled(r, 2) = double(rz);
      for (int k = 0; k < _nk; ++k) {
        auto kr         = rmesh_scaled(r, 0) * kmesh(k, 0) + rmesh_scaled(r, 1) * kmesh(k, 1) + rmesh_scaled(r, 2) * kmesh(k, 2);
        _T_k_to_r(r, k) = std::exp(-2i * M_PI * kr) * double(1. / _nk);
        _T_r_to_k(k, r) = std::exp(2i * M_PI * kr);
      }
    }

    _kmesh = kmesh.copy();
    dtensor<2> qmesh(kmesh.shape());
    for (int j = 0; j < _nk; ++j) {
      dtensor<1> ki = kmesh(0);
      dtensor<1> kj = kmesh(j);
      auto       kq = details::wrap(ki - kj);
      qmesh(j) << kq;
    }

    for (int i = 0; i < _nk; ++i) {
      dtensor<1> ki = kmesh(i);
      for (int j = 0; j < _nk; ++j) {
        dtensor<1> kj = kmesh(j);
        auto       kq = details::wrap(ki - kj);
        int        q  = details::find_pos(kq, qmesh);
        _q_ind(j, q)  = i;
        _q_ind2(i, j) = q;
      }
    }
  }

  /**
   * compute integral momenta using momentum conservation
   *
   * @param n - first integral momentum triplet
   * @return full 4 momenta for the current triplet
   */
  template <typename Symmetry>
  std::array<size_t, 4> brillouin_zone_utils<Symmetry>::momentum_conservation(const std::array<size_t, 3>& n) const {
    return {n[0], n[1], n[2], mom_cons(n[0], n[1], n[2])};
  };

  template <typename Symmetry>
  size_t brillouin_zone_utils<Symmetry>::mom_cons(size_t i, size_t j, size_t k) const {
    int q = _q_ind2(i, j);
    int l = _q_ind(k, q);
    return l;
  }

  template <typename Symmetry>
  template <typename tensor_t>
  void brillouin_zone_utils<Symmetry>::k_to_r(const tensor_t& in, tensor_t& out) const {
    size_t dim1  = in.shape()[0];
    size_t dim_k = std::accumulate(in.shape().begin() + 1, in.shape().end(), 1ul, std::multiplies<size_t>());
    size_t dim_r = std::accumulate(out.shape().begin() + 1, out.shape().end(), 1ul, std::multiplies<size_t>());
    assert(dim_k == dim_r);
    CMMatrixXcd f_k(in.data(), dim1, dim_k);
    MMatrixXcd  f_r(out.data(), dim1, dim_r);
    f_r = (_T_k_to_r * f_k).eval();
  }
  template <typename Symmetry>
  template <typename tensor_t>
  void brillouin_zone_utils<Symmetry>::r_to_k(const tensor_t& in, tensor_t& out) const {
    size_t dim1  = in.shape()[0];
    size_t dim_r = std::accumulate(in.shape().begin() + 1, in.shape().end(), 1ul, std::multiplies<size_t>());
    size_t dim_k = std::accumulate(out.shape().begin() + 1, out.shape().end(), 1ul, std::multiplies<size_t>());
    assert(dim_k == dim_r);
    CMMatrixXcd f_r(in.data(), dim1, dim_r);
    MMatrixXcd  f_k(out.data(), dim1, dim_k);
    f_k = (_T_r_to_k * f_r).eval();
  }

  template class brillouin_zone_utils<inv_symm_op>;

#ifndef FT_BZ_OP
#define FT_BZ_OP(N)                                                                           \
  template void brillouin_zone_utils<inv_symm_op>::k_to_r(const ztensor<N>& in, ztensor<N>& out) const; \
  template void brillouin_zone_utils<inv_symm_op>::r_to_k(const ztensor<N>& in, ztensor<N>& out) const;
#endif

  FT_BZ_OP(4)
  FT_BZ_OP(3)
  FT_BZ_OP(2)
  FT_BZ_OP(1)

}  // namespace green::symmetry