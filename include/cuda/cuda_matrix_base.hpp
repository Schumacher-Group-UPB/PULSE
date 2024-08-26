#pragma once

namespace PC3 {

static inline bool global_matrix_creation_log = true;
static inline bool global_matrix_transfer_log = true;

/**
 * @brief Base class for all device matrices. Mostly a gimmick to have a global
 * counter for the total amount of allocated device memory.
 */
class CUDAMatrixBase {
   public:
    static inline double global_total_device_mb = 0;
    static inline double global_total_device_mb_max = 0;
    static inline double global_total_host_mb = 0;
    static inline double global_total_host_mb_max = 0;
};

}