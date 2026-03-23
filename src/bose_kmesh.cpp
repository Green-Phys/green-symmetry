#include "green/symmetry/bose_kmesh.h"

namespace green::symmetry {
  bose_kmesh::bose_kmesh(const green::params::params& p) {
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
    in_file["symmetry/q/n_stars"] >> _n_stars;
    _stars.resize(_n_stars);
    for (size_t i = 0; i < _n_stars; ++i) {
      itensor<1> star_i;
      in_file["symmetry/q/stars/" + std::to_string(i)] >> star_i;
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
}