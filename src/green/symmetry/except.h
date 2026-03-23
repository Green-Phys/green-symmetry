#ifndef GREEN_SYMMETRY_EXCEPT_H
#define GREEN_SYMMETRY_EXCEPT_H

#include <cstdio>
#include <stdexcept>
#include <string>

namespace green::symmetry {

  static const std::string SYMMETRY_INPUT_MIN_VERSION = "1.0.0";

  class symmetry_incorrect_input_error : public std::runtime_error {
  public:
    explicit symmetry_incorrect_input_error(const std::string& what) : std::runtime_error(what) {}
  };

  class symmetry_outdated_input : public std::runtime_error {
  public:
    explicit symmetry_outdated_input(const std::string& what) : std::runtime_error(what) {}
  };

  /**
   * @brief Compare two version strings
   * 
   * @param v - version string to check
   * @return true if v >= SYMMETRY_INPUT_MIN_VERSION
   * @return false otherwise
   */
  bool CheckVersion(const std::string& v);

}  // namespace green::symmetry
#endif // GREEN_SYMMETRY_EXCEPT_H