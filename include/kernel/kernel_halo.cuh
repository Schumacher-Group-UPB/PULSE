#pragma once
#include "cuda/typedef.cuh"
#include "cuda/cuda_macro.cuh"
#include "kernel/kernel_index_overwrite.cuh"

namespace PC3::Kernel::Halo {

    template <typename T>
    PULSE_GLOBAL void full_grid_to_halo_grid(int i, Type::uint N_x, Type::uint N_y, Type::uint subgrids_x, Type::uint subgrid_N_x, Type::uint subgrid_N_y, Type::uint halo_size, T* fullgrid, T** subgrids) {
        GET_THREAD_INDEX( i, N_x*N_y );
        
        const Type::uint r = i / N_x;
        const Type::uint c = i % N_x;

        const Type::uint R = r / subgrid_N_y;
        const Type::uint C = c / subgrid_N_x;
        const Type::uint subgrid = R*subgrids_x + C;

        const Type::uint r_subgrid = halo_size + r % subgrid_N_y;
        const Type::uint c_subgrid = halo_size + c % subgrid_N_x;

        const Type::uint subgrid_with_halo = subgrid_N_x + 2*halo_size;
        // Now move value from subgrid to fullgrid
        subgrids[subgrid][r_subgrid*subgrid_with_halo + c_subgrid] = fullgrid[i];
    }

    template <typename T>
    PULSE_GLOBAL void halo_grid_to_full_grid(int i, Type::uint N_x, Type::uint N_y, Type::uint subgrids_x, Type::uint subgrid_N_x, Type::uint subgrid_N_y, Type::uint halo_size, T* fullgrid, T** subgrids) {
        GET_THREAD_INDEX( i, N_x*N_y );
        
        const Type::uint r = i / N_x;
        const Type::uint c = i % N_x;

        const Type::uint R = r / subgrid_N_y;
        const Type::uint C = c / subgrid_N_x;
        const Type::uint subgrid = R*subgrids_x + C;

        const Type::uint r_subgrid = halo_size + r % subgrid_N_y;
        const Type::uint c_subgrid = halo_size + c % subgrid_N_x;

        const Type::uint subgrid_with_halo = subgrid_N_x + 2*halo_size;
        // Now move value from subgrid to fullgrid
        fullgrid[i] = subgrids[subgrid][r_subgrid*subgrid_with_halo + c_subgrid];
    }

    // This kernel has execution range halo_num*subgrids_x*subgrids_y
    template <typename T>
    PULSE_GLOBAL void synchronize_halos(int i, Type::uint subgrids_x, Type::uint subgrids_y, Type::uint subgrid_N_x, Type::uint subgrid_N_y, Type::uint halo_size, Type::uint halo_num, bool periodic_boundary_x, bool periodic_boundary_y, bool twin_mode, T** subgrids_wf_plus, T** subgrids_wf_minus, T** subgrids_rv_plus, T** subgrids_rv_minus, int* subgrid_map) {
        GET_THREAD_INDEX( i, halo_num*subgrids_x*subgrids_y );

        const Type::uint sg = i / halo_num; // subgrid index from 0 to subgrids_x*subgrids_y.
        const Type::uint s = (i % halo_num)*6; // subgrid_map index from 0 to 6*len(subgrid_map)
    
        const int R = sg / subgrids_x;
        const int C = sg % subgrids_x;
        const Type::uint subgrid = R*subgrids_x + C;
    
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

        const Type::uint r_new = (R + dr) % subgrids_y;
        const Type::uint c_new = (C + dc) % subgrids_x;
        subgrids_wf_plus[subgrid][tr*(subgrid_N_x+2*halo_size) + tc] = subgrids_wf_plus[r_new*subgrids_x+c_new][fr*(subgrid_N_x+2*halo_size) + fc];
        subgrids_rv_plus[subgrid][tr*(subgrid_N_x+2*halo_size) + tc] = subgrids_rv_plus[r_new*subgrids_x+c_new][fr*(subgrid_N_x+2*halo_size) + fc];
        if (not twin_mode)
            return;
        subgrids_wf_minus[subgrid][tr*(subgrid_N_x+2*halo_size) + tc] = subgrids_wf_minus[r_new*subgrids_x+c_new][fr*(subgrid_N_x+2*halo_size) + fc];
        subgrids_rv_minus[subgrid][tr*(subgrid_N_x+2*halo_size) + tc] = subgrids_rv_minus[r_new*subgrids_x+c_new][fr*(subgrid_N_x+2*halo_size) + fc];
    }
} // namespace PC3::Kernel::Halo