#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "cuda/cuda_complex.cuh"
#include "cuda/cuda_macro.cuh"
#include "misc/escape_sequences.hpp"

namespace PC3 {

static inline bool global_matrix_creation_log = false;
static inline bool global_matrix_transfer_log = false;

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

/**
 * @brief CUDA Wrapper for a matrix
 * Handles memory management and host-device synchronization when
 * using the getDevicePtr() and getHostPtr() functions.
 *
 * @tparam T either real_number or complex_number
 */
template <typename T>
class CUDAMatrix : CUDAMatrixBase {
   private:
    unsigned int rows, cols;
    unsigned int total_size;
    std::string name;
    T* device_data = nullptr;
    std::unique_ptr<T[]> host_data;
    double size_in_mb;
    bool is_on_device;
    bool is_on_host;

    bool host_is_ahead;

   public:
    CUDAMatrix() {
        device_data = nullptr;
        host_data = nullptr;
        name = "unnamed";
        rows = 0;
        cols = 0;
        total_size = 0;
        is_on_device = false;
        is_on_host = false;
        host_is_ahead = false;
    };

    CUDAMatrix( CUDAMatrix& other ) : rows( other.rows ), cols( other.cols ), total_size( other.total_size ), name( other.name ), device_data( other.device_data ), host_data( other.host_data ) {
        other.total_size = 0;
        other.cols = 0;
        other.rows = 0;
        other.device_data = nullptr;
        other.host_data = nullptr;
    }
    CUDAMatrix( CUDAMatrix&& other ) : rows( other.rows ), cols( other.cols ), total_size( other.total_size ), name( other.name ), device_data( other.device_data ), host_data( other.host_data ) {
        other.total_size = 0;
        other.cols = 0;
        other.rows = 0;
        other.device_data = nullptr;
        other.host_data = nullptr;
    }

    CUDAMatrix( unsigned int rows, unsigned int cols, const std::string& name ) : rows( rows ), cols( cols ), name( name ) {
        construct( rows, cols, name );
    }

    CUDAMatrix( unsigned int rows, unsigned int cols, T* data, const std::string& name ) : CUDAMatrix( rows, cols, name ) {
        setTo( data );
    }

    CUDAMatrix( unsigned int root_size, const std::string& name ) : CUDAMatrix( root_size, root_size, name ){};
    CUDAMatrix( unsigned int root_size, T* data, const std::string& name ) : CUDAMatrix( root_size, root_size, data, name ){};

    bool isOnDevice() const {
        return is_on_device;
    }
    bool isOnHost() const {
        return is_on_host;
    }

    CUDAMatrix<T>& setTo( T* data ) {
        if ( not is_on_host ) {
            std::cout << EscapeSequence::RED << "Matrix '" << name << "' is not on host." << EscapeSequence::RESET << std::endl;
            return *this;
        }
        std::copy( data, data + total_size, host_data.get() );
        host_is_ahead = true;
        return *this;
    }

    void destroy_host() {
        if ( host_data == nullptr or rows == 0 or cols == 0 or total_size == 0 )
            return;
        global_total_host_mb -= size_in_mb;
        if ( global_matrix_creation_log )
            std::cout << EscapeSequence::GRAY << "Freeing " << rows << "x" << cols << " matrix '" << name << "' from host, total allocated host space: " << global_total_host_mb << " MB." << EscapeSequence::RESET << std::endl;
        host_data.reset();
        host_data = nullptr;
        is_on_host = false;
    }

    void destroy_device() {
        if ( device_data == nullptr or rows == 0 or cols == 0 or total_size == 0 )
            return;
        global_total_device_mb -= size_in_mb;
        if ( global_matrix_creation_log )
            std::cout << EscapeSequence::GRAY << "Freeing " << rows << "x" << cols << " matrix '" << name << "' from device, total allocated device space: " << global_total_device_mb << " MB." << EscapeSequence::RESET << std::endl;
        DEVICE_FREE( device_data, "free " );
        device_data = nullptr;
        is_on_device = false;
    }

    ~CUDAMatrix() {
        destroy_host();
        destroy_device();
    }

    CUDAMatrix<T>& constructDevice( unsigned int rows, unsigned int cols, const std::string& name ) {
        total_size = rows * cols;
        size_in_mb = total_size * sizeof( T ) / 1024.0 / 1024.0;
        global_total_device_mb += size_in_mb;
        global_total_device_mb_max = std::max( global_total_device_mb, global_total_device_mb_max );
        this->name = name;
        this->rows = rows;
        this->cols = cols;
        if ( global_matrix_creation_log )
            std::cout << EscapeSequence::GRAY << "Allocating " << size_in_mb << " MB for " << rows << "x" << cols << " device matrix '" << name << "', total allocated device space: " << global_total_device_mb << " MB." << EscapeSequence::RESET << std::endl;
        DEVICE_ALLOC( device_data, total_size * sizeof( T ), "malloc " );
        is_on_device = true;
        host_is_ahead = false;
        return *this;
    }
    CUDAMatrix<T>& constructDevice( unsigned int root_size, const std::string& name ) {
        return constructDevice( root_size, root_size, name );
    }

    CUDAMatrix<T>& constructHost( unsigned int rows, unsigned int cols, const std::string& name = "unnamed" ) {
        total_size = rows * cols;
        size_in_mb = total_size * sizeof( T ) / 1024.0 / 1024.0;
        global_total_host_mb += size_in_mb;
        global_total_host_mb_max = std::max( global_total_host_mb, global_total_host_mb_max );
        this->name = name;
        this->rows = rows;
        this->cols = cols;
        if ( global_matrix_creation_log )
            std::cout << EscapeSequence::GRAY << "Allocating " << size_in_mb << " MB for " << rows << "x" << cols << " host matrix '" << name << "', total allocated host space: " << global_total_host_mb << " MB." << EscapeSequence::RESET << std::endl;
        host_data = std::make_unique<T[]>( total_size );
        is_on_host = true;
        host_is_ahead = true;
        return *this;
    }
    CUDAMatrix<T>& constructHost( unsigned int root_size, const std::string& name = "unnamed" ) {
        return constructHost( root_size, root_size, name );
    }

    CUDAMatrix<T>& construct( unsigned int rows, unsigned int cols, const std::string& name = "unnamed" ) {
        constructHost( rows, cols, name );
        constructDevice( rows, cols, name );
        return *this;
    }
    CUDAMatrix<T>& construct( unsigned int root_size, const std::string& name = "unnamed" ) {
        return construct( root_size, root_size, name );
    }

    // Synchronize Host and Device. This functions usually doesn't have to be called manually by the user.
    CUDAMatrix<T>& hostToDeviceSync() {
        // If the matrix does not exist on device yet, create it from host parameters
        if ( not is_on_device and total_size > 0 )
            constructDevice( rows, cols, name );

        if ( global_matrix_transfer_log )
            std::cout << EscapeSequence::GRAY << "Copying " << rows << "x" << cols << " matrix to device matrix '" << name << "'" << EscapeSequence::RESET << std::endl;
        MEMCOPY_TO_DEVICE( device_data, host_data.get(), total_size * sizeof( T ), "memcopy host to device " );
        return *this;
    }

    // Synchronize Host and Device. This functions usually doesn't have to be called manually by the user.
    CUDAMatrix<T>& deviceToHostSync() {
        // If the matrix does not exist on host yet, create it from device parameters
        if ( not is_on_host and total_size > 0 )
            constructHost( rows, cols, name );

        if ( global_matrix_transfer_log )
            std::cout << EscapeSequence::GRAY << "Copying " << rows << "x" << cols << " matrix from device matrix '" << name << "'" << EscapeSequence::RESET << std::endl;
        MEMCOPY_FROM_DEVICE( host_data.get(), device_data, total_size * sizeof( T ), "memcpy device to host " );
        return *this;
    }

    void swap( CUDAMatrix<T>& other ) {
        if ( rows != other.rows or cols != other.cols )
            return;
        std::swap( device_data, other.device_data );
        std::swap( host_data, other.host_data );
    }

    enum class Direction {
        ROW,
        COLUMN
    };

    // Only Host Data is slicable
    std::vector<T> slice( const int start, const int size, Direction direction = Direction::ROW ) {
        std::unique_ptr<T[]> buffer_out = std::make_unique<T[]>( size );
        std::copy( host_data.get() + start, host_data.get() + start + size, buffer_out.get() );
        std::vector<T> out( buffer_out.get(), buffer_out.get() + size );
        return out;
    }

    std::vector<T> sliceDevice( const int start, const int size, Direction direction = Direction::ROW ) {
        std::unique_ptr<T[]> buffer_out = std::make_unique<T[]>( size );
        MEMCOPY_FROM_DEVICE( buffer_out.get(), device_data + start, size * sizeof( T ), "memcpy device to host buffer" );
        std::vector<T> out( buffer_out.get(), buffer_out.get() + size );
        return out;
    }

    inline unsigned int getTotalSize() const {
        return total_size;
    }

    // Device Pointer Getter
    inline T* getDevicePtr() {
        // If the host is ahead, synchronize first
        if ( host_is_ahead ) {
            hostToDeviceSync();
            host_is_ahead = false;
        }
        return device_data;
    }
    inline T* getHostPtr() {
        // If the device is ahead, synchronize first
        if ( not host_is_ahead ) {
            deviceToHostSync();
            host_is_ahead = true;
        }
        return host_data.get();
    }

    // Row, col and index getters for host matrix
    // The [] operator does synchronize the host and device memory
    inline T at( int index ) const {
        return getHostPtr()[index];
    }
    inline T at( int row, int column ) const {
        return getHostPtr()[row * rows + column];
    }
    inline T& at( int index ) {
        return getHostPtr()[index];
    }
    inline T& at( int row, int column ) {
        return getHostPtr()[row * rows + column];
    }
    inline T& operator[]( int index ) {
        return host_data.get()[index];
    }
    inline T operator()( int row, int column ) {
        return getHostPtr()[row * rows + column];
    }
};

} // namespace PC3