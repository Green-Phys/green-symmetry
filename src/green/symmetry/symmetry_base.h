#ifndef GREEN_SYMMETRY_BASE_H
#define GREEN_SYMMETRY_BASE_H

#include <stdexcept>
#include "common_defs.h"


namespace green::symmetry {

  /**
   * @brief Base class for mesh (k or q) symmetry operations
   * 
   * Provides common symmetry reduction and point mapping functionality shared between
   * k-point and q-point symmetries. Derived classes handle specific data loading and transformations.
   */
  class symmetry_base {
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
    const std::vector<long>& tr_conj_list() const { return _tr_conj_list; }
    
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
     * Get the appropriate operation (conjugate or identity) for a mesh point
     *
     * @tparam T - data type
     * @param idx - mesh point index
     * @return operation functor to apply
     */
    template <typename T>
    Op_t<T> op(size_t idx) const {
      return _tr_conj_list[idx] ? conj_op<T> : no_op<T>;
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
     * For a given value from reduced BZ and a mesh point from the full BZ,
     * determine the corresponding value in the full BZ
     *
     * @param val - value from reduced BZ
     * @param idx - mesh point to compute actual value for
     * @return corresponding value in the full BZ
     */
    std::complex<double> value(std::complex<double> val, size_t idx) const { 
      return op<std::complex<double>>(idx)(val);
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
    std::vector<long> _tr_conj_list;
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

} // namespace green::symmetry

#endif // GREEN_SYMMETRY_BASE_H
