#include <green/symmetry/integral_pair_map.h>

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <limits>

TEST_CASE("SG representative preload size is checked", "[integral-storage]") {
  using green::symmetry::integral_storage_bytes;
  REQUIRE(integral_storage_bytes(24, 10, 8, 16, 245760) == 245760);
  REQUIRE_THROWS_AS(integral_storage_bytes(24, 10, 8, 16, 245759), std::runtime_error);
  REQUIRE_THROWS_AS(
      integral_storage_bytes(std::numeric_limits<std::size_t>::max(), 10, 8, 16, std::numeric_limits<std::size_t>::max()),
      std::overflow_error);
}

TEST_CASE("SG whitened auxiliary maps preserve the interaction", "[integral-storage]") {
  using green::symmetry::validate_auxiliary_isometry;
  using complex                        = std::complex<double>;
  const std::array<complex, 4> unitary = {complex(0), complex(0, 1), complex(0, -1), complex(0)};
  REQUIRE_NOTHROW(validate_auxiliary_isometry(unitary.data(), 2));
  // Raw metric diag(1, 1e-16) versus diag(1, 4e-16) differs by only 3e-16,
  // but its whitened map diag(1, 1/2) changes an ERI contribution by 75%.
  const std::array<complex, 4> nonisometric = {complex(1), complex(0), complex(0), complex(.5)};
  REQUIRE_THROWS_AS(validate_auxiliary_isometry(nonisometric.data(), 2), std::runtime_error);
  REQUIRE_THROWS_AS(validate_auxiliary_isometry(nullptr, 2), std::runtime_error);
  REQUIRE_THROWS_AS(validate_auxiliary_isometry(unitary.data(), 0), std::runtime_error);
}
