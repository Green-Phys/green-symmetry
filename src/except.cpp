#include "green/symmetry/except.h"
#include <cstdio>

namespace green::symmetry {

bool CheckVersion(const std::string& v) {
    int major_Vin, minor_Vin, patch_Vin;
    int major_Vref, minor_Vref, patch_Vref;

    char suffixV[32] = "";
    char suffixM[32] = "";

    std::sscanf(v.c_str(), "%d.%d.%d%31s", &major_Vin, &minor_Vin, &patch_Vin, suffixV);
    std::sscanf(SYMMETRY_INPUT_MIN_VERSION.c_str(), "%d.%d.%d%31s", &major_Vref, &minor_Vref, &patch_Vref, suffixM);

    if (major_Vin != major_Vref) return major_Vin > major_Vref;
    if (minor_Vin != minor_Vref) return minor_Vin > minor_Vref;
    if (patch_Vin != patch_Vref) return patch_Vin > patch_Vref;

    // If numeric parts in version are all equal, do not worry about suffix
    // e.g., 0.2.4b10 has same integral format as 0.2.4
    return true;
}

} // namespace green::symmetry