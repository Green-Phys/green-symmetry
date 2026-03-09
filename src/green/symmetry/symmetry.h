/*
 * Copyright (c) 2020-2022 University of Michigan.
 *
 */

#ifndef GREEN_BZ_UTILS_H
#define GREEN_BZ_UTILS_H

#include "common_defs.h"
#include "except.h"

namespace green::symmetry {

  /**
   * @brief Base class for mesh (k or q) symmetry operations
   * 
   * Provides common symmetry reduction and point mapping functionality shared between
   * k-point and q-point symmetries. Derived classes handle specific data loading and transformations.
   */
  class mesh_symmetry {
  public:
    /**
     * @return number of points in full mesh BZ
     */
    size_t nk() const { return _nk; }

    /**
     * @return number of points in reduced mesh BZ
     */
    size_t ink() const { return _ink; }

    /**
     * @return scaled reciprocal mesh used for Fourier transforms
     */
    const dtensor<2>& mesh() const { return _mesh; }

    /**
     * @return vector that maps mesh point from the full BZ to a point in the reduced BZ
     */
    const std::vector<size_t>& full_to_reduced() const { return _full_to_reduced; }
    
    /**
     * @return vector that maps mesh point from the reduced BZ to a point in the full BZ
     */
    const std::vector<size_t>& reduced_to_full() const { return _reduced_to_full; }
    
    /**
     * @return list of flags for each mesh point in the full BZ telling whether we need to do complex conjugations
     * to get corresponding point in reduced BZ
     */
    const std::vector<long>& conj_list() const { return _conj_list; }
    
    /**
     * @return list of weight for each mesh point in the full BZ
     */
    const std::vector<double>& weight() const { return _weight; }
    
    /**
     * @return list of degenerate points (stars) for given irreducible point
     */
    const std::vector<long>& deg(size_t i) const { return star(i); }
    
    /**
     * For a given index of a point from the full BZ return an index of a corresponding point
     * from the reduced BZ
     *
     * @param idx - index of a point in the full BZ
     * @return index of a corresponding point in the reduced BZ
     */
    size_t reduced_point(size_t idx) const { return _full_to_reduced[idx]; }
    
    /**
     * For a given index of a point from the reduced BZ return an index in the full BZ
     *
     * @param iidx - index in the reduced BZ
     * @return index in the full BZ
     */
    size_t full_point(size_t iidx) const { return _reduced_to_full[iidx]; }

    /**
     * For a given value from reduced BZ and a mesh point from the full BZ,
     * determine the corresponding value in the full BZ
     *
     * @param val - value from reduced BZ
     * @param idx - mesh point to compute actual value for
     * @return corresponding value in the full BZ
     */
    std::complex<double> value(std::complex<double> val, size_t idx) const { 
      return _conj_list[idx] == 0 ? val : std::conj(val); 
    }

    /**
     * Get the appropriate operation (conjugate or identity) for a mesh point
     *
     * @tparam T - data type
     * @param idx - mesh point index
     * @return operation functor to apply
     */
    template <typename T>
    Op_t<T> op(size_t idx) const {
      return _conj_list[idx] ? conj_op<T> : no_op<T>;
    }
    
    /**
     * @brief Get the star of mesh point indices for given irreducible point
     * 
     * @param star_idx - index of the star/irreducible point
     * @return const std::vector<long>& - indices of equivalent mesh points in the full BZ
     */
    const std::vector<long>& star(size_t star_idx) const {
      return _stars[star_idx];
    }

    /**
     * @brief Get the number of stars
     * 
     * @return const size_t& 
     */
    const size_t& n_stars() const {
      return _n_stars;
    }

  protected:
    // Mapping of mesh point from full BZ to reduced BZ
    std::vector<size_t> _full_to_reduced;
    // Mapping of mesh point from reduced BZ to full BZ
    std::vector<size_t> _reduced_to_full;
    // weight of the mesh point in reduced BZ
    std::vector<double> _weight;
    // conjugate list
    std::vector<long> _conj_list;
    // Stars information (equivalent mesh points for each irreducible point)
    std::vector<std::vector<long>> _stars;
    size_t _n_stars = 0;
    // mesh metadata
    size_t _nk = 0;
    size_t _ink = 0;
    dtensor<2> _mesh;

    /**
     * Build internal full-to-reduced and degeneracy mappings from loaded reduced-to-full index list.
     * Called by derived class constructors after loading basic mesh data.
     *
     * @param index - vector of full BZ indices corresponding to each full BZ point
     */
    void build_mappings(const std::vector<long>& index) {
      _full_to_reduced.resize(index.size());
      long ir_idx;
      for (size_t i = 0; i < index.size(); ++i) {
        ir_idx = -1;
        for (size_t j = 0; j < _reduced_to_full.size(); ++j) {
          if (_reduced_to_full[j] == index[i]) {
            ir_idx = j;
            break;
          }
        }
        if (ir_idx < 0) {
          throw std::runtime_error("Can not find corresponding point in the reduced BZ");
        }
        _full_to_reduced[i] = ir_idx;
      }
    }
  };


  /**
   * @deprecated Use kpoint_symmetry instead. Kept for backward compatibility.
   */
  using inv_symm_op = kpoint_symmetry;


  class kpoint_symmetry : public mesh_symmetry {
  public:
    kpoint_symmetry(const green::params::params& p) {
      size_t               nao;
      std::vector<long>    index;
      green::h5pp::archive in_file(p["input_file"], "r");
      in_file["grid/k/nk"] >> _nk;
      in_file["grid/k/ink"] >> _ink;
      in_file["grid/k/_mesh_scaled"] >> _mesh;
      in_file["grid/k/weight"] >> _weight;
      in_file["grid/k/index"] >> index;
      in_file["grid/k/ir_list"] >> _reduced_to_full;
      in_file["grid/k/conj_list"] >> _conj_list;
      in_file["grid/pairs/num_kpair_stored"] >> _num_kpair_stored;
      in_file["grid/pairs/conj_pairs_list"] >> _conj_kpair_list;
      in_file["grid/pairs/trans_pairs_list"] >> _trans_kpair_list;
      in_file["grid/pairs/kpair_irre_list"] >> _kpair_irre_list;
      in_file["params/nao"] >> nao;
      
      // Check input version to determine if symmetry data should be present
      if (in_file.has_attribute("__green_version__")) {
        std::string version = in_file.get_attribute<std::string>("__green_version__");
        if (!CheckVersion(version)) {
          in_file.close();
          throw symmetry_outdated_input("Input file version " + version + " is too old. Minimum required version is " + SYMMETRY_INPUT_MIN_VERSION);
        }
      }
      
      // Read trnansformation matrices
      k_sym_transform_ao_.resize(index.size(), nao, nao);
      in_file["grid/symmetry/k/k_sym_transform_ao"] >> k_sym_transform_ao_;
      // Information about stars of k-points
      in_file["grid/symmetry/k/n_stars"] >> _n_stars;
      _stars.resize(_n_stars);
      for (size_t i = 0; i < _n_stars; ++i) {
        itensor<1> star_i;
        in_file["grid/symmetry/k/stars/" + std::to_string(i)] >> star_i;
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
    const std::vector<long>& conj_list() const { return _conj_list; }

    size_t                   num_kpair_stored() const { return _num_kpair_stored; }

    const std::vector<long>& conj_kpair_list() const { return _conj_kpair_list; }

    const std::vector<long>& trans_kpair_list() const { return _trans_kpair_list; }

    const std::vector<long>& kpair_irre_list() const { return _kpair_irre_list; }

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
    std::complex<double> value(std::complex<double> val, size_t k) const { return _conj_list[k] == 0 ? val : std::conj(val); }

    template <typename T>
    Op_t<T> op(size_t k) const {
      return _conj_list[k] ? conj_op<T> : no_op<T>;
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
      size_t nao = k_sym_transform_ao_.shape()[1];
      CMMatrixXcd U_k_dummy(k_sym_transform_ao_.data() + k * nao * nao, nao, nao);
      U_k_dummy.resize(nao, nao);
      U_k = U_k_dummy.cast<prec>();
    }

  private:
    // k-space symmetry transform in AO basis
    ztensor<3> k_sym_transform_ao_;

    // k-pairs information
    std::vector<long> _conj_kpair_list;
    std::vector<long> _trans_kpair_list;
    std::vector<long> _kpair_irre_list;
    size_t            _num_kpair_stored;
  };


  /**
   * @brief Symmetry operations for q-points (momentum transfers)
   * 
   * Handles symmetry reductions and transformations for q-mesh (BZ of momentum transfers).
   * Loads q-mesh grid and symmetry operators from green-mbtools HDF5 output.
   */
  class qpoint_symmetry : public mesh_symmetry {
  public:
    qpoint_symmetry(const green::params::params& p) {
      size_t               naux;
      std::vector<long>    index;
      green::h5pp::archive in_file(p["input_file"], "r");
      
      // Read q-mesh info
      in_file["grid/q/nq"] >> _nk;
      in_file["grid/q/inq"] >> _ink;
      in_file["grid/q/mesh_scaled"] >> _mesh;
      in_file["grid/q/weight"] >> _weight;
      in_file["grid/q/index"] >> index;
      in_file["grid/q/ir_list"] >> _reduced_to_full;
      in_file["grid/q/conj_list"] >> _conj_list;
      
      // dimension of atomic orbital and auxiliary basis
      in_file["params/NQ"] >> naux;
      
      // Check input version to determine if symmetry data should be present
      if (in_file.has_attribute("__green_version__")) {
        std::string version = in_file.get_attribute<std::string>("__green_version__");
        if (!CheckVersion(version)) {
          in_file.close();
          throw symmetry_outdated_input("Input file version " + version + " is too old. Minimum required version is " + SYMMETRY_INPUT_MIN_VERSION);
        }
      }
      
      // j2c metric basis transformation for q-points
      q_sym_transform_j2c_.resize(index.size(), naux, naux);
      in_file["symmetry/q/k_sym_transform_j2c"] >> q_sym_transform_j2c_;
        
      // P0 polarization transformation (in j2c^{-1/2} basis)
      q_sym_transform_p0_.resize(index.size(), naux, naux);
      in_file["symmetry/q/k_sym_transform_p0"] >> q_sym_transform_p0_;
      
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
      size_t naux = q_sym_transform_j2c_.shape()[1];
      CMMatrixXcd U_q_dummy(q_sym_transform_j2c_.data() + q * naux * naux, naux, naux);
      U_q_dummy.resize(naux, naux);
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
      size_t naux = q_sym_transform_p0_.shape()[1];
      CMMatrixXcd U_q_p0_dummy(q_sym_transform_p0_.data() + q * naux * naux, naux, naux);
      U_q_p0_dummy.resize(naux, naux);
      U_q_p0 = U_q_p0_dummy.cast<prec>();
    }
    
  private:
    // q-space symmetry transform in j2c metric basis
    ztensor<3> q_sym_transform_j2c_;
    // q-space symmetry transform for P0 (in j2c^{-1/2} basis)
    ztensor<3> q_sym_transform_p0_;
  };


  template <typename Symmetry = kpoint_symmetry>
  class brillouin_zone_utils {
  public:
    brillouin_zone_utils(const green::params::params& p);

    /**
     * @return object describing system symmetries
     */
    const Symmetry& symmetry() const { return _symmetry; }
    /**
     * @return number of k-points in the full first Brillouin zone
     */
    size_t nk() const { return _nk; }

    /**
     * @return number of k-points in the reduced first Brillouin zone
     */
    size_t ink() const { return _ink; }

    /**
     * @return number of AOs based on input file
     */
    size_t nao() const { return _nao; }

    /**
     * @return number of spin-orbitals based on input file
     */
    size_t nso() const { return _nso; }

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
     * @param val - array to be projected
     * @return new array on the full first BZ that corresponds to input array via defined symmetry relations
     */
    template <typename T>
    auto ibz_to_full(const green::ndarray::ndarray<T, 3>& val) const {
      assert(val.shape()[0] == _ink);
      std::array<size_t, 3> new_shape(val.shape());
      new_shape[0] = _nk;
      green::ndarray::ndarray<std::remove_const_t<T>, 3> ret(new_shape);

      if (!_X2C){
        for (size_t k = 0; k < _nk; ++k) {
          size_t ik = _symmetry.reduced_point(k);
          std::transform(val(ik).begin(), val(ik).end(), ret(k).begin(),
                         [this, k](const T& item) { return _symmetry.template op<T>(k)(item); });
        }
      } else {
        // for X2C, spin-flip is necessary: X(-k) = Spin-flip [X(k).conj()]
        for (size_t k = 0; k < _nk; ++k) {
          size_t ik = _symmetry.reduced_point(k);
          if (_symmetry.conj_list()[k]) {
            for (size_t i = 0; i < _nao; ++i) {
              for (size_t j = 0; j < _nao; ++j) {
                  ret(k, i, j) = std::conj(val(ik, _nao + i, _nao + j));
                  ret(k, _nao + i, _nao + j) = std::conj(val(ik, i, j));
                  ret(k, i, _nao + j) = -1.0 * std::conj(val(ik, _nao + i, j));
                  ret(k, _nao + i, j) = -1.0 * std::conj(val(ik, i, _nao + j));
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
      assert(val.shape()[0] == _nk);
      std::array<size_t, D> new_shape(val.shape());
      new_shape[0] = _ink;
      green::ndarray::ndarray<std::remove_const_t<T>, D> ret(new_shape);
      for (size_t ik = 0; ik < _ink; ++ik) {
        size_t k = _symmetry.full_point(ik);
        ret(ik) << val(k);
      }
      return ret;
    } // LCOV_EXCL_LINE

  private:
    Symmetry   _symmetry;
    size_t     _nk;
    size_t     _ink;
    size_t     _nao;
    size_t     _nso;
    bool       _X2C;

    double     _nkpw;

    itensor<2> _q_ind;
    itensor<2> _q_ind2;

    dtensor<2> _kmesh;

    MatrixXcd  _T_k_to_r;
    MatrixXcd  _T_r_to_k;

    size_t     mom_cons(size_t i, size_t j, size_t k) const;
  };
}  // namespace green::symmetry

#endif  // GREEN_BZ_UTILS_H
