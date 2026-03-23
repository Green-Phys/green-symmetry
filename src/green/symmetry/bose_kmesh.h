#ifndef GREEN_QPOINT_SYMMETRY_H
#define GREEN_QPOINT_SYMMETRY_H

#include "except.h"
#include "symmetry_base.h"
#include "shape_utils.h"

namespace green::symmetry {

  /**
   * @brief Symmetry information for q-point mesh, which is usually obtained from k1-k2 pairs in the k-mesh.
   * 
   * Loads q-mesh grid and symmetry operators from green-mbtools HDF5 output.
   * Handles symmetry reductions and transformations between the full and reduced q-mesh.
   */
  class bose_kmesh : public symmetry_base {
  public:
    bose_kmesh(const green::params::params& p);

    /**
     * @brief Get the q-space transformation matrix in j2c basis
     * 
     * @tparam prec 
     * @param U_q transformation matrix to be filled
     * @param q q-point index in the full BZ
     */
    template <typename prec>
    const void q_sym_transform_j2c(MatrixX<prec>& U_q, size_t q) const {
      CMMatrixXcd U_q_dummy(_q_sym_transform_j2c.data() + q * _naux * _naux, _naux, _naux);
      U_q_dummy.resize(_naux, _naux);
      U_q = U_q_dummy.cast<prec>();
    }
    
    /**
     * @brief Get the q-space P0 transformation matrix (in metric basis)
     * 
     * @tparam prec 
     * @param U_q_p0 transformation matrix to be filled
     * @param q q-point index in the full BZ
     */
    template <typename prec>
    const void q_sym_transform_p0(MatrixX<prec>& U_q_p0, size_t q) const {
      CMMatrixXcd U_q_p0_dummy(_q_sym_transform_p0.data() + q * _naux * _naux, _naux, _naux);
      U_q_p0_dummy.resize(_naux, _naux);
      U_q_p0 = U_q_p0_dummy.cast<prec>();
    }

    /**
     * Obtain 'value' of array at a given k-point in the full BZ by applying
     * symmetry transformation to the value at the corresponding point in the reduced BZ
     *
     * @param val
     * @param q
     * @return
     */
    template <typename T>
    MatrixX<std::remove_const_t<T>> value_J2C(const green::ndarray::ndarray<T, 3>& val, size_t q) const {
      using ST = std::remove_const_t<T>;
      green::symmetry::validate_shape(
        {val.shape()[0], val.shape()[1], val.shape()[2]},
        {_ink, _naux, _naux},
        "value_J2C"
      );
      size_t iq = reduced_point(q);
      auto   val_iq = val(iq);

      CMMatrixX<ST> U_q(_q_sym_transform_j2c.data() + q * _naux * _naux, _naux, _naux);
      CMMatrixX<ST> val_iq_m(val_iq.data(), _naux, _naux);
      MatrixX<ST>   U_q_cast = U_q.template cast<ST>();
      MatrixX<ST>   val_iq_cast = val_iq_m.template cast<ST>();
      MatrixX<ST>   val_tran = U_q_cast * val_iq_cast * U_q_cast.adjoint();

      if (_tr_conj_list[q] != 0) {
        val_tran = val_tran.conjugate();
      }
      return val_tran;
    }

    /**
     * Obtain 'value' of array at a given k-point in the full BZ by applying
     * symmetry transformation to the value at the corresponding point in the reduced BZ
     *
     * @param val
     * @param q
     * @return
     */
    template <typename T>
    MatrixX<std::remove_const_t<T>> value_P0(const green::ndarray::ndarray<T, 3>& val, size_t q) const {
      using ST = std::remove_const_t<T>;
      green::symmetry::validate_shape(
        {val.shape()[0], val.shape()[1], val.shape()[2]},
        {_ink, _naux, _naux},
        "value_P0"
      );
      size_t iq = reduced_point(q);
      auto   val_iq = val(iq);

      CMMatrixX<ST> U_q(_q_sym_transform_p0.data() + q * _naux * _naux, _naux, _naux);
      CMMatrixX<ST> val_iq_m(val_iq.data(), _naux, _naux);
      MatrixX<ST>   U_q_cast = U_q.template cast<ST>();
      MatrixX<ST>   val_iq_cast = val_iq_m.template cast<ST>();
      MatrixX<ST>   val_tran = U_q_cast * val_iq_cast * U_q_cast.adjoint();

      if (_tr_conj_list[q] != 0) {
        val_tran = val_tran.conjugate();
      }
      return val_tran;
    }

    size_t naux() const { return _naux; }

  private:
    // Number of auxiliary AOs (dimension of q-space operators)
    size_t _naux;
    // q-space symmetry transform in j2c metric basis
    ztensor<3> _q_sym_transform_j2c;
    // q-space symmetry transform for P0 (in j2c^{-1/2} basis)
    ztensor<3> _q_sym_transform_p0;
  };
} // namespace green::symmetry

#endif // GREEN_QPOINT_SYMMETRY_H