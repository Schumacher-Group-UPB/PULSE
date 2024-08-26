#pragma once
#include "cuda/typedef.cuh"
#include "cuda/cuda_macro.cuh"
#include "kernel/kernel_index_overwrite.cuh"

namespace PC3::Kernel::Halo {

    template <typename T>
    PULSE_GLOBAL void full_grid_to_halo_grid(int i, size_t N_x, size_t N_y, size_t subgrids_x, size_t subgrid_N_x, size_t subgrid_N_y, size_t halo_size, T* fullgrid, T** subgrids) {
        GET_THREAD_INDEX( i, N_x*N_y );
        
        const size_t r = i / N_x;
        const size_t c = i % N_x;

        const size_t R = r / subgrid_N_y;
        const size_t C = c / subgrid_N_x;
        const size_t subgrid = R*subgrids_x + C;

        const size_t r_subgrid = halo_size + r % subgrid_N_y;
        const size_t c_subgrid = halo_size + c % subgrid_N_x;

        const size_t subgrid_with_halo = subgrid_N_x + 2*halo_size;
        // Now move value from subgrid to fullgrid
        subgrids[subgrid][r_subgrid*subgrid_with_halo + c_subgrid] = fullgrid[i];
    }

    template <typename T>
    PULSE_GLOBAL void halo_grid_to_full_grid(int i, size_t N_x, size_t N_y, size_t subgrids_x, size_t subgrid_N_x, size_t subgrid_N_y, size_t halo_size, T* fullgrid, T** subgrids) {
        GET_THREAD_INDEX( i, N_x*N_y );
        
        const size_t r = i / N_x;
        const size_t c = i % N_x;

        const size_t R = r / subgrid_N_y;
        const size_t C = c / subgrid_N_x;
        const size_t subgrid = R*subgrids_x + C;

        const size_t r_subgrid = halo_size + r % subgrid_N_y;
        const size_t c_subgrid = halo_size + c % subgrid_N_x;

        const size_t subgrid_with_halo = subgrid_N_x + 2*halo_size;
        // Now move value from subgrid to fullgrid
        fullgrid[i] = subgrids[subgrid][r_subgrid*subgrid_with_halo + c_subgrid];
    }

    // This kernel has execution range halo_num*subgrids_x*subgrids_y
    template <typename T>
    PULSE_GLOBAL void synchronize_halos(int i, size_t subgrids_x, size_t subgrids_y, size_t subgrid_N_x, size_t subgrid_N_y, size_t halo_size, size_t halo_num, bool periodic_boundary_x, bool periodic_boundary_y, T** subgrids, int* subgrid_map) {
        GET_THREAD_INDEX( i, halo_num );
        
        const size_t sg = i / halo_num; // subgrid index from 0 to subgrids_x*subgrids_y.
        const size_t s = i % halo_num; // subgrid_map index from 0 to 6*halo_num
    
        const int R = sg / subgrids_y;
        const int C = sg % subgrids_y;
        const size_t subgrid = R*subgrids_x + C;
    
        const auto dr = subgrid_map[s];
        const auto dc = subgrid_map[s+1];
        const auto fr = subgrid_map[s+2];
        const auto fc = subgrid_map[s+3];
        const auto tr = subgrid_map[s+4];
        const auto tc = subgrid_map[s+5];
        
        // Subgrid remains zero if the boundary condition is not periodic
        if (not periodic_boundary_x and (C+dc < 0 or C+dc >= subgrids_x))
                return;
        if (not periodic_boundary_y and (R+dr < 0 or R+dr >= subgrids_y))
                return;
        const size_t r_new = (R + dr) % subgrids_y;
        const size_t c_new = (C + dc) % subgrids_x;
        subgrids[subgrid][tr*(subgrid_N_x+2*halo_size) + tc] = subgrids[r_new*subgrids_x+c_new][fr*(subgrid_N_x+2*halo_size) + fc];
    }
} // namespace PC3::Kernel::Halo