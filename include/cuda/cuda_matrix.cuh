#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_macro.cuh"
#include "misc/escape_sequences.hpp"

namespace PC3 {

static inline bool global_matrix_creation_log = true;
static inline bool global_matrix_transfer_log = false;

// Forward declaration of HostMatrix
template <typename T>
class HostMatrix;

/**
 * @brief CUDA Wrapper for a matrix of either real or complex numbers
 * Handles memory management.
 *
 * @tparam T either real_number or complex_number
 */
template <typename T> //, typename = std::enable_if_t<std::is_same_v<T, real_number> || std::is_same_v<T, complex_number>>>
class CUDAMatrix {
   private:
    unsigned int rows, cols; // One-Dimensional Size of size x size matrix
    unsigned int total_size;
    std::string name;
    T* device_data = nullptr;
    double size_in_mb;

   public:
    static inline double total_mb = 0;
    static inline double total_mb_max = 0;

    CUDAMatrix(){
        device_data = nullptr;
        name = "unnamed";
        rows = 0;
        cols = 0;
        total_size = 0;
    };

    CUDAMatrix( CUDAMatrix& other ) : rows( other.rows ), cols( other.cols ), total_size( other.total_size ), name( other.name ), device_data( other.device_data ) {
        other.total_size = 0;
        other.cols = 0;
        other.rows = 0;
        other.device_data = nullptr;
    }
    CUDAMatrix( CUDAMatrix&& other ) : rows( other.rows ), cols( other.cols ), total_size( other.total_size ), name( other.name ), device_data( other.device_data ) {
        other.total_size = 0;
        other.cols = 0;
        other.rows = 0;
        other.device_data = nullptr;
    }

    CUDAMatrix( unsigned int rows, unsigned int cols, const std::string& name ) : rows( rows ), cols( cols ), name( name ) {
        construct( rows, cols, name );
    }

    CUDAMatrix( unsigned int rows, unsigned int cols, T* data, const std::string& name ) : CUDAMatrix( rows, cols, name ) {
        setTo( data );
    }

    CUDAMatrix( unsigned int root_size, const std::string& name ) : CUDAMatrix( root_size, root_size, name ){};
    CUDAMatrix( unsigned int root_size, T* data, const std::string& name ) : CUDAMatrix( root_size, root_size, data, name ){};

    ~CUDAMatrix() {
        if ( device_data == nullptr or rows == 0 or cols == 0 or total_size == 0 )
            return;
        total_mb -= size_in_mb;
        if ( global_matrix_creation_log )
            std::cout << EscapeSequence::GREY << "Freeing " << rows << "x" << cols << " matrix '" << name << "' from device, total allocated device space: " << total_mb << " MB." << EscapeSequence::RESET << std::endl;
        DEVICE_FREE( device_data, "free " );
        device_data = nullptr;
    }

    CUDAMatrix<T>& construct( unsigned int rows, unsigned int cols, const std::string& name ) {
        total_size = rows * cols;
        size_in_mb = total_size * sizeof( T ) / 1024.0 / 1024.0;
        total_mb += size_in_mb;
        total_mb_max = std::max( total_mb, total_mb_max );
        this->name = name;
        this->rows = rows;
        this->cols = cols;
        if ( global_matrix_creation_log )
            std::cout << EscapeSequence::GREY << "Allocating " << size_in_mb << " MB for " << rows << "x" << cols << " matrix '" << name << "', total allocated device space: " << total_mb << " MB." << EscapeSequence::RESET << std::endl;
        DEVICE_ALLOC( device_data, total_size * sizeof( T ), "malloc " );
        return *this;
    }
    CUDAMatrix<T>& construct( unsigned int root_size, const std::string& name ) {
        return construct( root_size, root_size, name );
    }

    void setTo( T* host_data ) {
        if ( global_matrix_transfer_log )
            std::cout << EscapeSequence::GREY << "Copying " << rows << "x" << cols << " matrix to device matrix '" << name << "'" << EscapeSequence::RESET << std::endl;
        MEMCOPY_TO_DEVICE( device_data, host_data, total_size * sizeof( T ), "memcopy host to device " );
    }
    void fromHost( HostMatrix<T>& host_matrix ) {
        setTo( host_matrix.get() );
    }

    void copyTo( T* host_data ) {
        if ( global_matrix_transfer_log )
            std::cout << EscapeSequence::GREY << "Copying " << rows << "x" << cols << " matrix from device matrix '" << name << "'" << EscapeSequence::RESET << std::endl;
        MEMCOPY_FROM_DEVICE( host_data, device_data, total_size * sizeof( T ), "memcpy device to host " );
    }
    void toHost( HostMatrix<T>& host_matrix ) {
        copyTo( host_matrix.get() );
    }

    void swap( CUDAMatrix<T>& other ) {
        if ( rows != other.rows or cols != other.cols )
            return;
        swap_symbol( device_data, other.device_data );
    }

    enum class Direction {
        ROW,
        COLUMN
    };

    // TODO: implement direction
    std::vector<T> slice( const int start, const int size, Direction direction = Direction::ROW ) {
        std::unique_ptr<T[]> buffer_out = std::make_unique<T[]>( size );
        MEMCOPY_FROM_DEVICE( buffer_out.get(), device_data + start, size * sizeof( T ), "memcpy device to host buffer" );
        std::vector<T> out( buffer_out.get(), buffer_out.get() + size );
        return out;
    }

    CUDA_HOST_DEVICE unsigned int getTotalSize() const {
        return total_size;
    }

    // Data Access is device only, since this is a device matrix.
    // If one wants to access the data on the host, one has to copy it first
    // into a HostMatrix instance.

    // Device Pointer Getter
    inline CUDA_HOST_DEVICE T* get() const {
        return device_data;
    }

    // Row, col and index getters
    inline CUDA_HOST_DEVICE T at( int index ) const {
        return device_data[index];
    }
    inline CUDA_HOST_DEVICE T at( int row, int column ) const {
        return device_data[row * rows + column];
    }
    inline CUDA_HOST_DEVICE T& at( int index ) {
        return device_data[index];
    }
    inline CUDA_HOST_DEVICE T& at( int row, int column ) {
        return device_data[row * rows + column];
    }
    inline CUDA_HOST_DEVICE T& operator[]( int index ) {
        return device_data[index];
    }
    inline CUDA_HOST_DEVICE T operator()( int row, int column ) {
        return device_data[row * rows + column];
    }
};

template <typename T> //, typename = std::enable_if_t<std::is_same_v<T, real_number> || std::is_same_v<T, complex_number>>>
class HostMatrix {
   private:
    unsigned int rows, cols; // One-Dimensional Size of size x size matrix
    unsigned int total_size;
    std::string name;
    std::unique_ptr<T[]> host_data;
    double size_in_mb;

   public:
    static inline double total_mb = 0;
    static inline double total_mb_max = 0;

    HostMatrix() {
        host_data = nullptr;
        name = "unnamed";
        rows = 0;
        cols = 0;
        total_size = 0;
    };

    HostMatrix( HostMatrix& other ) : rows( other.rows ), cols( other.cols ), total_size( other.total_size ), name( other.name ), host_data( other.host_data ) {
        other.total_size = 0;
        other.cols = 0;
        other.rows = 0;
        other.host_data.reset(); // TODO does this release/erase the data? Do i have to use a shared_ptr?
    }
    HostMatrix( HostMatrix&& other ) : rows( other.rows ), cols( other.cols ), total_size( other.total_size ), name( other.name ), host_data( other.host_data ) {
        other.total_size = 0;
        other.cols = 0;
        other.rows = 0;
        other.host_data.reset();
    }

    HostMatrix( unsigned int rows, unsigned int cols, const std::string& name ) : rows( rows ), cols( cols ), name( name ) {
        construct( rows, cols, name );
    }

    HostMatrix( unsigned int rows, unsigned int cols, T* data, const std::string& name ) : HostMatrix( rows, cols, name ) {
        setTo( data );
    }

    HostMatrix( unsigned int root_size, const std::string& name ) : HostMatrix( root_size, root_size, name ){};
    HostMatrix( unsigned int root_size, T* data, const std::string& name ) : HostMatrix( root_size, root_size, data, name ){};

    ~HostMatrix() {
        total_mb -= size_in_mb;
        if ( global_matrix_creation_log )
            std::cout << EscapeSequence::GREY << "Freeing " << rows << "x" << cols << " matrix '" << name << "' from device, total allocated host space: " << total_mb << " MB." << EscapeSequence::RESET << std::endl;
        host_data.reset();
    }

    HostMatrix<T>& construct( unsigned int rows, unsigned int cols, const std::string& name = "unnamed" ) {
        total_size = rows * cols;
        size_in_mb = total_size * sizeof( T ) / 1024.0 / 1024.0;
        total_mb += size_in_mb;
        total_mb_max = std::max( total_mb, total_mb_max );
        this->name = name;
        this->rows = rows;
        this->cols = cols;
        if ( global_matrix_creation_log )
            std::cout << EscapeSequence::GREY << "Allocating " << size_in_mb << " MB for " << rows << "x" << cols << " matrix '" << name << "', total allocated host space: " << total_mb << " MB." << EscapeSequence::RESET << std::endl;
        host_data = std::make_unique<T[]>( total_size );
        return *this;
    }
    HostMatrix<T>& construct( unsigned int root_size, const std::string& name = "unnamed" ) {
        return construct( root_size, root_size, name );
    }

    void setTo( T* data ) {
        if ( global_matrix_transfer_log )
            std::cout << EscapeSequence::GREY << "Copying " << rows << "x" << cols << " matrix to device matrix '" << name << "'" << EscapeSequence::RESET << std::endl;
        std::copy( data, data + total_size, host_data.get() );
    }
    void fromDevice( CUDAMatrix<T>& device_matrix ) {
        // setTo(device_matrix.get());
        device_matrix.copyTo( host_data.get() );
    }

    void copyTo( T* host_data ) {
        if ( global_matrix_transfer_log )
            std::cout << EscapeSequence::GREY << "Copying " << rows << "x" << cols << " matrix from device matrix '" << name << "'" << EscapeSequence::RESET << std::endl;
        std::copy( this->host_data.get(), this->host_data.get() + total_size, host_data );
    }
    void toDevice( CUDAMatrix<T>& device_matrix ) {
        // copyTo(device_matrix.get());
        device_matrix.setTo( host_data.get() );
    }

    void swap( HostMatrix<T>& other ) {
        if ( rows != other.rows or cols != other.cols )
            return;
        swap_symbol( host_data, other.host_data );
    }

    enum class Direction {
        ROW,
        COLUMN
    };

    std::vector<T> slice( const int start, const int size, Direction direction = Direction::ROW ) {
        std::unique_ptr<T[]> buffer_out = std::make_unique<T[]>( size );
        std::copy( host_data.get() + start, host_data.get() + start + size, buffer_out.get() );
        std::vector<T> out( buffer_out.get(), buffer_out.get() + size );
        return out;
    }

    inline unsigned int getTotalSize() const {
        return total_size;
    }

    // Device Pointer Getter
    inline T* get() const {
        if ( total_size == 0 )
            std::cout << EscapeSequence::YELLOW << "Warning: accessing empty host matrix '" << name << "'." << EscapeSequence::RESET << std::endl;
        return host_data.get();
    }

    // Row, col and index getters
    inline T at( int index ) const {
        return host_data.get()[index];
    }
    inline T at( int row, int column ) const {
        return host_data.get()[row * rows + column];
    }
    inline T& at( int index ) {
        return host_data.get()[index];
    }
    inline T& at( int row, int column ) {
        return host_data.get()[row * rows + column];
    }
    inline T& operator[]( int index ) {
        return host_data.get()[index];
    }
    inline T operator()( int row, int column ) {
        return host_data.get()[row * rows + column];
    }
};

} // namespace PC3