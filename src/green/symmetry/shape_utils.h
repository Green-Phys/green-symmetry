#ifndef GREEN_SYMMETRY_SHAPE_UTILS_H
#define GREEN_SYMMETRY_SHAPE_UTILS_H

#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>

namespace green::symmetry {

  inline std::string shape_to_string(const std::vector<size_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      oss << shape[i];
      if (i + 1 < shape.size()) oss << ", ";
    }
    oss << "]";
    return oss.str();
  }
  
  inline void validate_shape(const std::vector<size_t>& shape_in, const std::vector<size_t>& shape_expected, const char* func_name) {
    if (shape_in != shape_expected) {
      throw std::invalid_argument(
        std::string(func_name) + " expects shape " + shape_to_string(shape_expected) + ", but got " + shape_to_string(shape_in)
      );
    }
  }

} // namespace green::symmetry

#endif // GREEN_SYMMETRY_SHAPE_UTILS_H
