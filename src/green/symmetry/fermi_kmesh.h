#ifndef GREEN_KPOINT_SYMMETRY_H
#define GREEN_KPOINT_SYMMETRY_H

#include "except.h"
#include "symmetry_base.h"

namespace green::symmetry {

  /**
   * @brief Symmetry information for k-point mesh.
   * 
   * Loads k-mesh grid and symmetry operators from green-mbtools HDF5 output.
   * Handles symmetry reductions and transformations between the full and reduced k-mesh.
   */
  class fermi_kmesh : public symmetry_base {
  public:
    fermi_kmesh(const green::params::params& p) {
      std::vector<long>    index;
      green::h5pp::archive in_file(p["input_file"], "r");
      in_file["symmetry/k/nk"] >> _nk;
      in_file["symmetry/k/ink"] >> _ink;
      in_file["symmetry/k/mesh_scaled"] >> _mesh;
      in_file["symmetry/k/weight_ibz"] >> _weight;
      in_file["symmetry/k/bz2ibz"] >> index;
      in_file["symmetry/k/ibz2bz"] >> _reduced_to_full;
      in_file["symmetry/k/tr_conj"] >> _tr_conj_list;
      in_file["symmetry/pairs/num_kpair_stored"] >> _num_kpair_stored;
      in_file["symmetry/pairs/conj_pairs_list"] >> _conj_kpair_list;
      in_file["symmetry/pairs/trans_pairs_list"] >> _trans_kpair_list;
      in_file["symmetry/pairs/kpair_irre_list"] >> _kpair_irre_list;
      in_file["params/nao"] >> _nao;
      in_file["params/nso"] >> _nso;

      // Check input version to determine if symmetry data should be present
      if (in_file.has_attribute("__green_version__")) {
        std::string version = in_file.get_attribute<std::string>("__green_version__");
        if (!CheckVersion(version)) {
          in_file.close();
          throw symmetry_outdated_input("Input file version " + version + " is too old. Minimum required version is " + SYMMETRY_INPUT_MIN_VERSION);
        }
      }
      
      // Read trnansformation matrices
      _k_sym_transform_ao.resize(index.size(), _nso, _nso);
      in_file["symmetry/k/k_sym_transform_ao"] >> _k_sym_transform_ao;
      // Information about stars of k-points
      in_file["symmetry/k/n_stars"] >> _n_stars;
      _stars.resize(_n_stars);
      for (size_t i = 0; i < _n_stars; ++i) {
        itensor<1> star_i;
        in_file["symmetry/k/stars/" + std::to_string(i)] >> star_i;
        // Convert itensor<1> to std::vector<long>
        _stars[i] = std::vector<long>(star_i.data(), star_i.data() + star_i.size());
      }
      
      in_file.close();
      build_mappings(index);
    }

    /**
     * @return list of flags for each k-point in the full first BZ telling whether we need to du complex conjugations
     * to get corresponding point in reduced BZ
     */
    const std::vector<long>& tr_conj_list() const { return _tr_conj_list; }

    size_t                   num_kpair_stored() const { return _num_kpair_stored; }

    const std::vector<long>& conj_kpair_list() const { return _conj_kpair_list; }

    const std::vector<long>& trans_kpair_list() const { return _trans_kpair_list; }

    const std::vector<long>& kpair_irre_list() const { return _kpair_irre_list; }
    
    size_t nao() const { return _nao; }

    size_t nso() const { return _nso; }

    /**
     * @return list of weight for each k-point in the full first BZ (w = 2 if k-point has exact mapping to reduced BZ, 0
     * otherwise)
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
    std::complex<double> value(std::complex<double> val, size_t k) const { return _tr_conj_list[k] == 0 ? val : std::conj(val); }

    template <typename T>
    Op_t<T> op(size_t k) const {
      return _tr_conj_list[k] ? conj_op<T> : no_op<T>;
    }

    /**
     * @brief Get the AO-basis rotation matrix to transform from IBZ to index k in the full BZ
     * 
     * @tparam prec 
     * @param U_k transformation matrix to be filled
     * @param k k-point index in the full BZ
     */
    template <typename prec>
    const void k_sym_transform_ao(MatrixX<prec>& U_k, size_t k) const {
      CMMatrixXcd U_k_dummy(_k_sym_transform_ao.data() + k * _nso * _nso, _nso, _nso);
      U_k_dummy.resize(_nso, _nso);
      U_k = U_k_dummy.cast<prec>();
    }

    /**
     * Obtain 'value' of array at a given k-point in the full BZ by applying
     * symmetry transformation to the value at the corresponding point in the reduced BZ
     *
     * @param val
     * @param k
     * @return
     */
    template <typename T>
    MatrixX<std::remove_const_t<T>> value_AO(const green::ndarray::ndarray<T, 3>& val, size_t k) const {
      using ST = std::remove_const_t<T>;
      assert(val.shape()[0] == _ink);
      assert(val.shape()[1] == _nso);
      assert(val.shape()[2] == _nso);
      size_t ik = reduced_point(k);
      auto   val_ik = val(ik);

      CMMatrixX<ST> U_k(_k_sym_transform_ao.data() + k * _nso * _nso, _nso, _nso);
      CMMatrixX<ST> val_ik_m(val_ik.data(), _nso, _nso);
      MatrixX<ST>   U_k_cast      = U_k.template cast<ST>();
      MatrixX<ST>   val_ik_cast   = val_ik_m.template cast<ST>();
      MatrixX<ST>   val_tran      = U_k_cast * val_ik_cast * U_k_cast.adjoint();

      if (_tr_conj_list[k] != 0) {
        // Anti-unitary mapping (time reversal): (U * val * U^dagger)^*.
        val_tran = val_tran.conjugate();
      }
      return val_tran;
    }

  private:
    // k-space symmetry transform in AO basis
    ztensor<3> _k_sym_transform_ao;

    // number of atomic orbitals
    size_t _nao;
    // number of spin orbitals -- practical dimension of G / Sigma / etc.
    size_t _nso;

    // k-pairs information
    std::vector<long> _conj_kpair_list;
    std::vector<long> _trans_kpair_list;
    std::vector<long> _kpair_irre_list;
    size_t            _num_kpair_stored;
  };

} // namespace green::symmetry

#endif // GREEN_KPOINT_SYMMETRY_H