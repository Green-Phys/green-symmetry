#ifndef GREEN_QPOINT_SYMMETRY_H
#define GREEN_QPOINT_SYMMETRY_H

#include "except.h"
#include "symmetry_base.h"

namespace green::symmetry {

  /**
   * @brief Symmetry information for q-point mesh, which is usually obtained from k1-k2 pairs in the k-mesh.
   * 
   * Loads q-mesh grid and symmetry operators from green-mbtools HDF5 output.
   * Handles symmetry reductions and transformations between the full and reduced q-mesh.
   */
  class bose_kmesh : public symmetry_base {
  public:
    bose_kmesh(const green::params::params& p) {
      std::vector<long>    index;
      green::h5pp::archive in_file(p["input_file"], "r");
      dtensor<4> q_sym_transform_j2c_tmp;
      dtensor<4> q_sym_transform_p0_tmp;
      
      // Read q-mesh info
      in_file["symmetry/q/nq"] >> _nk;
      in_file["symmetry/q/inq"] >> _ink;
      in_file["symmetry/q/mesh_scaled"] >> _mesh;
      in_file["symmetry/q/weight_ibz"] >> _weight;
      in_file["symmetry/q/bz2ibz"] >> index;
      in_file["symmetry/q/ibz2bz"] >> _reduced_to_full;
      in_file["symmetry/q/tr_conj"] >> _tr_conj_list;
      
      // dimension of atomic orbital and auxiliary basis
      in_file["params/NQ"] >> _naux;
      
      // Check input version to determine if symmetry data should be present
      if (in_file.has_attribute("__green_version__")) {
        std::string version = in_file.get_attribute<std::string>("__green_version__");
        if (!CheckVersion(version)) {
          in_file.close();
          throw symmetry_outdated_input("Input file version " + version + " is too old. Minimum required version is " + SYMMETRY_INPUT_MIN_VERSION);
        }
      }
      
      // Information about stars of k-points
      in_file["symmetry/k/n_stars"] >> _n_stars;
      _stars.resize(_n_stars);
      for (size_t i = 0; i < _n_stars; ++i) {
        itensor<1> star_i;
        in_file["symmetry/k/stars/" + std::to_string(i)] >> star_i;
        // Convert itensor<1> to std::vector<long>
        _stars[i] = std::vector<long>(star_i.data(), star_i.data() + star_i.size());
      }
      
      // j2c metric basis transformation for q-points
      _q_sym_transform_j2c.resize(index.size(), _naux, _naux);
      in_file["symmetry/q/k_sym_transform_j2c"] >> _q_sym_transform_j2c;
        
      // P0 polarization transformation (in j2c^{-1/2} basis)
      _q_sym_transform_p0.resize(index.size(), _naux, _naux);
      in_file["symmetry/q/k_sym_transform_p0"] >> _q_sym_transform_p0;
      
      in_file.close();
      build_mappings(index);
    }

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
    template <typename T, size_t D>
    auto value_J2C(const green::ndarray::ndarray<T, D>& val, size_t q) const {
      assert(val.shape()[0] == _ink);
      // Read transformation matrix
      size_t iq = reduced_point(q);
      CMMatrixX<T> U_q(_q_sym_transform_j2c.data() + q * _naux * _naux, _naux, _naux);
      // transform and return
      auto val_tran = U_q * val(reduced_point(q)) * U_q.adjoint();
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
    template <typename T, size_t D>
    auto value_P0(const green::ndarray::ndarray<T, D>& val, size_t q) const {
      assert(val.shape()[0] == _ink);
      // Read transformation matrix
      size_t iq = reduced_point(q);
      CMMatrixX<T> U_q(_q_sym_transform_p0.data() + q * _naux * _naux, _naux, _naux);
      // transform and return
      auto val_tran = U_q * val(reduced_point(q)) * U_q.adjoint();
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