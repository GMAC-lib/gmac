#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
#include <string>
#include <sstream>

#include <gmac/cuda.h>

typedef std::map<GmacAcceleratorType, std::string> AccStrings;
static AccStrings accTypeString;

static GmacAcceleratorType accTypes [] = { GMAC_ACCELERATOR_TYPE_UNKNOWN,
                                            GMAC_ACCELERATOR_TYPE_CPU,
                                            GMAC_ACCELERATOR_TYPE_GPU,
                                            GMAC_ACCELERATOR_TYPE_ACCELERATOR };

static void init_map()
{
    accTypeString.insert(AccStrings::value_type(GMAC_ACCELERATOR_TYPE_UNKNOWN,     "GMAC_ACCELERATOR_TYPE_UNKNOWN"));
    accTypeString.insert(AccStrings::value_type(GMAC_ACCELERATOR_TYPE_CPU,         "GMAC_ACCELERATOR_TYPE_CPU"));
    accTypeString.insert(AccStrings::value_type(GMAC_ACCELERATOR_TYPE_GPU,         "GMAC_ACCELERATOR_TYPE_GPU"));
    accTypeString.insert(AccStrings::value_type(GMAC_ACCELERATOR_TYPE_ACCELERATOR, "GMAC_ACCELERATOR_TYPE_ACCELERATOR"));
}

static std::string get_type_string(GmacAcceleratorType type)
{
    std::string type_string;
    for (unsigned i = 0; i < 4; i++) {
        GmacAcceleratorType t = accTypes[i];
        if ((type & t) != 0) {
            if (type_string.size() > 0) {
                type_string += " | ";
            }
            type_string.append(accTypeString[t]);
        }
    }

    return type_string;
}

static std::string get_dim_sizes_string(unsigned dims, const size_t *maxSizes)
{
    std::string dim_sizes_string;
    for (unsigned d = 0; d < dims; d++) {
        std::stringstream ss;
        ss << maxSizes[d];
        if (dim_sizes_string.size() > 0) {
            dim_sizes_string.append(", ");
        }
        dim_sizes_string.append(ss.str());
    }
    return dim_sizes_string;
}

int main(int argc, char *argv[])
{
    GmacAcceleratorInfo info;

    init_map();

    for (unsigned i = 0; i < gmacGetNumberOfAccelerators(); i++) {
        assert(gmacGetAcceleratorInfo(i, &info) == gmacSuccess);
        fprintf(stdout, "Accelerator %u/%u\n", i + 1, gmacGetNumberOfAccelerators());

        fprintf(stdout, "- name: %s\n", info.acceleratorName);
        fprintf(stdout, "- vendor: %s\n", info.vendorName);
        fprintf(stdout, "- vendor id: %u\n", info.vendorId);
        fprintf(stdout, "- type: %s\n", get_type_string(info.acceleratorType).c_str());
        fprintf(stdout, "- available: %u\n", info.isAvailable);

        fprintf(stdout, "- compute units: %u\n", info.computeUnits);
        fprintf(stdout, "- max dimensions: %u\n", info.maxDimensions);
        fprintf(stdout, "- max sizes: %s\n", get_dim_sizes_string(info.maxDimensions, info.maxSizes).c_str());
        fprintf(stdout, "- max work group size: "FMT_SIZE"\n", info.maxWorkGroupSize);

        fprintf(stdout, "- global mem size: "FMT_SIZE"\n", info.globalMemSize);
        fprintf(stdout, "- local mem size: "FMT_SIZE"\n", info.localMemSize);
        fprintf(stdout, "- cache mem size: Not available\n");
        fprintf(stdout, "- dirver: %u\n", info.driverMajor);

        fprintf(stdout, "\n");
    }

    return 0;
}
