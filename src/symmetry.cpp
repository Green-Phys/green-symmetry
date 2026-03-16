/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#include <green/params/params.h>

#include <complex>

#include "green/symmetry/symmetry.h"

using namespace std::literals::complex_literals;

namespace green::symmetry {

  brillouin_zone_utils::brillouin_zone_utils(const green::params::params& p) : _k_symmetry(p), _q_symmetry(p),
    _kq_map(_k_symmetry, _q_symmetry) {
    const dtensor<2>&    kmesh = _k_symmetry.mesh();
    size_t nk = _k_symmetry.nk();
    size_t ink = _k_symmetry.ink();
    size_t nq = _q_symmetry.nk();

    _X2C = false;
    size_t nao = _k_symmetry.nao();
    size_t nso = _k_symmetry.nso();
    if (nao != nso) {_X2C = true;}

    _T_k_to_r.resize(nk, nk);
    _T_r_to_k.resize(nk, nk);
    _nkpw   = 1.0 / double(nk);
    _nqpw   = 1.0 / double(nq);

    int nkx = std::cbrt(double(nk));
    assert(nkx * nkx * nkx == nk);
    dtensor<2> rmesh_scaled(nk, 3);
    // TODO: This is hard-coded for (nk, nk, nk) type of k-mesh; it would not work for 2D systems.
    for (int r = 0; r < nk; ++r) {
      int rx             = r / (nkx * nkx);
      int ry             = (r / nkx) % nkx;
      int rz             = r % nkx;
      rmesh_scaled(r, 0) = double(rx);
      rmesh_scaled(r, 1) = double(ry);
      rmesh_scaled(r, 2) = double(rz);
      for (int k = 0; k < nk; ++k) {
        auto kr         = rmesh_scaled(r, 0) * kmesh(k, 0) + rmesh_scaled(r, 1) * kmesh(k, 1) + rmesh_scaled(r, 2) * kmesh(k, 2);
        _T_k_to_r(r, k) = std::exp(-2i * M_PI * kr) * _nkpw;
        _T_r_to_k(k, r) = std::exp(2i * M_PI * kr);
      }
    }
  }

  template <typename tensor_t>
  void brillouin_zone_utils::k_to_r(const tensor_t& in, tensor_t& out) const {
    size_t dim1  = in.shape()[0];
    size_t dim_k = std::accumulate(in.shape().begin() + 1, in.shape().end(), 1ul, std::multiplies<size_t>());
    size_t dim_r = std::accumulate(out.shape().begin() + 1, out.shape().end(), 1ul, std::multiplies<size_t>());
    assert(dim_k == dim_r);
    CMMatrixXcd f_k(in.data(), dim1, dim_k);
    MMatrixXcd  f_r(out.data(), dim1, dim_r);
    f_r = (_T_k_to_r * f_k).eval();
  }
  template <typename tensor_t>
  void brillouin_zone_utils::r_to_k(const tensor_t& in, tensor_t& out) const {
    size_t dim1  = in.shape()[0];
    size_t dim_r = std::accumulate(in.shape().begin() + 1, in.shape().end(), 1ul, std::multiplies<size_t>());
    size_t dim_k = std::accumulate(out.shape().begin() + 1, out.shape().end(), 1ul, std::multiplies<size_t>());
    assert(dim_k == dim_r);
    CMMatrixXcd f_r(in.data(), dim1, dim_r);
    MMatrixXcd  f_k(out.data(), dim1, dim_k);
    f_k = (_T_r_to_k * f_r).eval();
  }

#ifndef FT_BZ_OP
#define FT_BZ_OP(N)                                                                           \
  template void brillouin_zone_utils::k_to_r(const ztensor<N>& in, ztensor<N>& out) const; \
  template void brillouin_zone_utils::r_to_k(const ztensor<N>& in, ztensor<N>& out) const;
#endif

  FT_BZ_OP(4)
  FT_BZ_OP(3)
  FT_BZ_OP(2)
  FT_BZ_OP(1)

}  // namespace green::symmetry