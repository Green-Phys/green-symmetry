#include "green/symmetry/fermi_kmesh.h"

namespace green::symmetry {
  fermi_kmesh::fermi_kmesh(const green::params::params& p) {
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
    if (in_file.is_data("symmetry/k/nk_list")) {
      itensor<1> nkl(3);
      in_file["symmetry/k/nk_list"] >> nkl;
      _nk_list = {(int)nkl(0), (int)nkl(1), (int)nkl(2)};
    } else {
      int nk_cbrt = std::lround(std::cbrt(double(_nk)));
      if (nk_cbrt * nk_cbrt * nk_cbrt == (int)_nk) {
        _nk_list = {nk_cbrt, nk_cbrt, nk_cbrt};
      } else {
        int nk_sqrt = std::lround(std::sqrt(double(_nk)));
        if (nk_sqrt * nk_sqrt == (int)_nk) {
          _nk_list = {nk_sqrt, nk_sqrt, 1};
        } else {
          throw symmetry_incorrect_input_error("k-mesh must be nk^3 (3D) or nk^2 x 1 (2D); "
                                               "or regenerate input with params/nk_list.");
        }
      }
    }

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
}