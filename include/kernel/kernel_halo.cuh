#pragma once
#include "cuda/typedef.cuh"
#include "cuda/cuda_macro.cuh"
#include "kernel/kernel_index_overwrite.cuh"

namespace PC3::Kernel::Halo {

template <typename T>
PULSE_GLOBAL void full_grid_to_halo_grid( int i, Type::uint32 N_c, Type::uint32 N_r, Type::uint32 subgrids_columns, Type::uint32 subgrid_N_c, Type::uint32 subgrid_N_r,
                                          Type::uint32 halo_size, T* fullgrid, T** subgrids ) {
    GET_THREAD_INDEX( i, N_c * N_r );

    const Type::uint32 r = i / N_c;
    const Type::uint32 c = i % N_c;

    const Type::uint32 R = r / subgrid_N_r;
    const Type::uint32 C = c / subgrid_N_c;
    const Type::uint32 subgrid = R * subgrids_columns + C;

    const Type::uint32 r_subgrid = halo_size + r % subgrid_N_r;
    const Type::uint32 c_subgrid = halo_size + c % subgrid_N_c;

    const Type::uint32 subgrid_with_halo = subgrid_N_c + 2 * halo_size;
    // Now move value from subgrid to fullgrid
    subgrids[subgrid][r_subgrid * subgrid_with_halo + c_subgrid] = fullgrid[i];
}

template <typename T>
PULSE_GLOBAL void halo_grid_to_full_grid( int i, Type::uint32 N_c, Type::uint32 N_r, Type::uint32 subgrids_columns, Type::uint32 subgrid_N_c, Type::uint32 subgrid_N_r,
                                          Type::uint32 halo_size, T* fullgrid, T** subgrids ) {
    GET_THREAD_INDEX( i, N_c * N_r );

    const Type::uint32 r = i / N_c;
    const Type::uint32 c = i % N_c;

    const Type::uint32 R = r / subgrid_N_r;
    const Type::uint32 C = c / subgrid_N_c;
    const Type::uint32 subgrid = R * subgrids_columns + C;

    const Type::uint32 r_subgrid = halo_size + r % subgrid_N_r;
    const Type::uint32 c_subgrid = halo_size + c % subgrid_N_c;

    const Type::uint32 subgrid_with_halo = subgrid_N_c + 2 * halo_size;
    // Now move value from subgrid to fullgrid
    fullgrid[i] = subgrids[subgrid][r_subgrid * subgrid_with_halo + c_subgrid];
}

template <typename T>
PULSE_DEVICE PULSE_INLINE void __synchronize_halo( Type::uint32 subgrid_to, Type::uint32 index_to, Type::uint32 subgrid_from, Type::uint32 index_from,
                                                   T** current_subgridded_matrix ) {
    current_subgridded_matrix[subgrid_to][index_to] = current_subgridded_matrix[subgrid_from][index_from];
}

template <typename T>
PULSE_GLOBAL void synchronize_halos( int i, Type::uint32 subgrids_columns, Type::uint32 subgrids_rows, Type::uint32 subgrid_N_c, Type::uint32 subgrid_N_r, Type::uint32 halo_size,
                                     Type::uint32 halo_num, bool periodic_boundary_x, bool periodic_boundary_y, int* subgrid_map, T** current_subgridded_matrix ) {
    GET_THREAD_INDEX( i, halo_num * subgrids_columns * subgrids_rows );
    
    const Type::uint32 sg = i / halo_num;        // subgrid index from 0 to subgrids_columns*subgrids_rows.
    const Type::uint32 s = ( i % halo_num ) * 6; // subgrid_map index from 0 to 6*len(subgrid_map)

    const int R = sg / subgrids_columns;
    const int C = sg % subgrids_columns;
    const Type::uint32 subgrid = R * subgrids_columns + C;

    const auto dr = subgrid_map[s];
    const auto dc = subgrid_map[s + 1];
    const auto fr = subgrid_map[s + 2];
    const auto fc = subgrid_map[s + 3];
    const auto tr = subgrid_map[s + 4];
    const auto tc = subgrid_map[s + 5];

    // Subgrid remains zero if the boundary condition is not periodic
    if ( !periodic_boundary_x && ( C + dc < 0 || C + dc >= subgrids_columns ) )
        return;
    if ( !periodic_boundary_y && ( R + dr < 0 || R + dr >= subgrids_rows ) )
        return;

    const Type::uint32 r_new = ( R + dr ) % subgrids_rows;
    const Type::uint32 c_new = ( C + dc ) % subgrids_columns;

    __synchronize_halo( subgrid, tr * ( subgrid_N_c + 2 * halo_size ) + tc, r_new * subgrids_columns + c_new, fr * ( subgrid_N_c + 2 * halo_size ) + fc, current_subgridded_matrix );
}
} // namespace PC3::Kernel::Halo