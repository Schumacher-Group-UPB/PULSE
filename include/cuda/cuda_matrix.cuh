#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "cuda/typedef.cuh"
#include "cuda/cuda_macro.cuh"
#include "misc/escape_sequences.hpp"
#include "cuda/cuda_matrix_base.hpp"

namespace PC3 {

/**
 * @brief CUDA Wrapper for a matrix
 * Handles memory management and host-device synchronization when
 * using the getDevicePtr() and getHostPtr() functions.
 *
 * @tparam T either Type::real or Type::complex
 */
template <typename T>
class CUDAMatrix : CUDAMatrixBase {
   private:
    // Rows and Cols of the Matrix. These can be different, so rows =/= cols
    unsigned int rows, cols;
    // The total Size of the Matrix = rows*cols. Used for allocation purposes.
    unsigned int total_size;
    // Name of the Matrix. Mostly used for debugging purposes.
    std::string name;

    // Device Vector. When using nvcc, this is a thrust::device_vector. When using gcc, this is a std::vector
    Type::device_vector<T> device_data;
    // Host Vector. When using nvcc, this is a thrust::host_vector. When using gcc, this is a std::vector
    Type::host_vector<T> host_data;

    // Size of this Matrix. Mostly used for debugging and the Summary    
    double size_in_mb;

    // On Device and On Host flags. Matrices always have both a host and a device pointer, but host and device
    // Memory can be individually reserved. This class will manage the transfer and synchronization between
    // Host and Device arrays.
    bool is_on_device;
    bool is_on_host;
    bool host_is_ahead;

   public:
    CUDAMatrix() {
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
    }
    CUDAMatrix( CUDAMatrix&& other ) : rows( other.rows ), cols( other.cols ), total_size( other.total_size ), name( other.name ), device_data( other.device_data ), host_data( other.host_data ) {
        other.total_size = 0;
        other.cols = 0;
        other.rows = 0;
    }

    CUDAMatrix( unsigned int rows, unsigned int cols, const std::string& name ) : rows( rows ), cols( cols ), name( name ) {
        construct( rows, cols, name );
    }

    CUDAMatrix( unsigned int rows, unsigned int cols, T* data, const std::string& name ) : CUDAMatrix( rows, cols, name ) {
        setTo( data );
    }

    CUDAMatrix( unsigned int root_size, const std::string& name ) : CUDAMatrix( root_size, root_size, name ){};
    CUDAMatrix( unsigned int root_size, T* data, const std::string& name ) : CUDAMatrix( root_size, root_size, data, name ){};

    /*
        @return: bool - True if the matrix has memory allocated on the device.
    */
    bool isOnDevice() const {
        return is_on_device;
    }

    /*
        @return: bool - True if the matrix as memory allocated on the host.
    */
    bool isOnHost() const {
        return is_on_host;
    }

    /**
     * @return: std::string - The name of this matrix.
    */
    std::string getName() const {
        return name;
    }

    /**
     * Sets the host data of this matrix to the input.
     * @param data: Type::host_vector<T> Source vector. Only source vectors with matching types can be copied to the host data of the matrix.
     * @return: ptr to this object to enable chaining of these operations.
    */
    CUDAMatrix<T>& setTo( const Type::host_vector<T>& data ) {
        // If this matrix is not on host, we cannot set its host data. 
        if ( not is_on_host ) {
            std::cout << EscapeSequence::RED << "Matrix '" << name << "' is not on host." << EscapeSequence::RESET << std::endl;
            return *this;
        }
        // Set the host data to the current data. This works fine with both std::vector and thrust::vector
        host_data = data;
        // The host data has been updated, so the host is now ahead.
        host_is_ahead = true;
        // Log this action
        if (global_matrix_transfer_log)
            std::cout << EscapeSequence::GRAY << "Copied " << data.size() << " elements to '" << getName() << "'" << std::endl; 
        // Return this pointer
        return *this;
    }
    
    /**
     * Sets the host data of this matrix to the other matrix' data
     * @param other: Other CUDAMatrix of same size
    */
    CUDAMatrix<T>& setTo( CUDAMatrix<T>& other ) {
        // If this matrix is not on host, we cannot set its host data. 
        if ( not is_on_host or not other.isOnHost()) {
            std::cout << EscapeSequence::RED << "Matrix '" << name << "' or Matrix '" << other.getName() << "' is not on host." << EscapeSequence::RESET << std::endl;
            return *this;
        }
        // Set the host data to the current data. This works fine with both std::vector and thrust::vector
        host_data = other.getHostVector();
        // The host data has been updated, so the host is now ahead.
        host_is_ahead = true;
        // Log this action
        if (global_matrix_transfer_log)
            std::cout << EscapeSequence::GRAY << "Copied '" << other.getName() << "' to '" << getName() << "'" << std::endl; 
        // Return this pointer
        return *this;
    }

    /**
     * Destroys the host data by clearing the host_data vector.
     * This also sets is_on_host to false, so the host matrix has to be reconstructed to be used again.
     * @return: ptr to this object
     */
    CUDAMatrix<T>& destroy_host() {
        // If the host data is already destroyed, do nothing
        if ( host_data.empty() or rows == 0 or cols == 0 or total_size == 0 )
            return *this;
        // Subtract the current size of this host matrix from the global matrix size
        global_total_host_mb -= size_in_mb;
        // Log this action. Mostly for simple debugging.
        if ( global_matrix_creation_log )
            std::cout << EscapeSequence::GRAY << "Freeing " << rows << "x" << cols << " matrix '" << name << "' from host, total allocated host space: " << global_total_host_mb << " MB." << EscapeSequence::RESET << std::endl;
        // Clear the Host data
        host_data.clear();
        // And make sure this matrix is not flagged as on host.
        is_on_host = false;
        // Return this pointer
        return *this;
    }

    /**
     * Destroys the device data by clearing the device_vector.
     * This also sets is_on_device to false, so the device matrix has to be reconstructed to be used again.
    */
    CUDAMatrix<T>& destroy_device() {
        // If the device data is already destroyed, do nothing
        if ( device_data.empty() or rows == 0 or cols == 0 or total_size == 0 )
            return *this;
        // Subtract the current size of this device matrix from the global matrix size
        global_total_device_mb -= size_in_mb;
        // Log this action. Mostly for simple debugging.
        if ( global_matrix_creation_log )
            std::cout << EscapeSequence::GRAY << "Freeing " << rows << "x" << cols << " matrix '" << name << "' from device, total allocated device space: " << global_total_device_mb << " MB." << EscapeSequence::RESET << std::endl;
        // Clear the Device Data. This calls cudaFree when using nvcc and std::vector's .clear() when using gcc
        device_data.clear();
        // Make sure this matrix is not flagged as on device
        is_on_device = false;
        // Return this pointer
        return *this;
    }

    /**
     * Destructor. Destroys both the host and the device Matrix.
    */
    ~CUDAMatrix() {
        destroy_host();
        destroy_device();
    }

    /**
     * Constructs the Device Matrix vector. This function only allocates the memory and does not copy any data.
     * @return: ptr to this matrix.
    */
    CUDAMatrix<T>& constructDevice( unsigned int rows, unsigned int cols, const std::string& name ) {
        // Calculate the total size of this matrix as well as its size in bytes
        total_size = rows * cols;
        size_in_mb = total_size * sizeof( T ) / 1024.0 / 1024.0;
        // Add the size to the global counter for the device sizes and update the maximum encountered memory size
        global_total_device_mb += size_in_mb;
        global_total_device_mb_max = std::max( global_total_device_mb, global_total_device_mb_max );
        // Set the name, the rows and the cols of this object to the parameters passed.
        this->name = name;
        this->rows = rows;
        this->cols = cols;
        // Log this action.
        if ( global_matrix_creation_log )
            std::cout << EscapeSequence::GRAY << "Allocating " << size_in_mb << " MB for " << rows << "x" << cols << " device matrix '" << name << "', total allocated device space: " << global_total_device_mb << " MB." << EscapeSequence::RESET << std::endl;
        // Reserve space on the device. When using nvcc, this allocates device memory on the GPU using thrust. When using gcc, this allocates memory for the CPU using std::vector.
        device_data.resize( total_size );
        // This matrix is now on device
        is_on_device = true;
        // And the Device Matrix is now ahead of the host matrix
        host_is_ahead = false;
        // Return pointer to this
        return *this;
    }

    /**
     * Alias for constructDevice( rows, cols, name ) using root_size = rows*cols
    */
    CUDAMatrix<T>& constructDevice( unsigned int root_size, const std::string& name ) {
        return constructDevice( root_size, root_size, name );
    }

    /**
     * Constructs the Host Matrix vector. This function only allocates the memory and does not copy any data.
     * @return: ptr to this matrix.
    */
    CUDAMatrix<T>& constructHost( unsigned int rows, unsigned int cols, const std::string& name = "unnamed" ) {
        // Calculate the total size of this matrix as well as its size in bytes
        total_size = rows * cols;
        size_in_mb = total_size * sizeof( T ) / 1024.0 / 1024.0;
        // Add the size to the global counter for the host sizes and update the maximum encountered memory size
        global_total_host_mb += size_in_mb;
        global_total_host_mb_max = std::max( global_total_host_mb, global_total_host_mb_max );
        // Set the name, the rows and the cols of this object to the parameters passed.
        this->name = name;
        this->rows = rows;
        this->cols = cols;
        // Log this action.
        if ( global_matrix_creation_log )
            std::cout << EscapeSequence::GRAY << "Allocating " << size_in_mb << " MB for " << rows << "x" << cols << " host matrix '" << name << "', total allocated host space: " << global_total_host_mb << " MB." << EscapeSequence::RESET << std::endl;
        // Reserve space on the device. When using nvcc, this allocates device memory on the GPU using thrust. When using gcc, this allocates memory for the CPU using std::vector.
        host_data.resize( total_size );
        // This Matrix is now on the host
        is_on_host = true;
        // The Host Matrix is now ahead of the device matrix
        host_is_ahead = true;
        // Return pointer to this
        return *this;
    }

    /**
     * Alias for constructHost( rows, cols, name ) using root_size = rows*cols
    */
    CUDAMatrix<T>& constructHost( unsigned int root_size, const std::string& name = "unnamed" ) {
        return constructHost( root_size, root_size, name );
    }

    /**
     * Constructs both the Host and the Device Matrices.
     * This overwrites the name and the sizes, but we dont care.
    */
    CUDAMatrix<T>& construct( unsigned int rows, unsigned int cols, const std::string& name = "unnamed" ) {
        constructDevice( rows, cols, name );
        constructHost( rows, cols, name );
        return *this;
    }

    /**
     * Alias for construct( rows, cols, name ) using root_size = rows*cols
    */
    CUDAMatrix<T>& construct( unsigned int root_size, const std::string& name = "unnamed" ) {
        return construct( root_size, root_size, name );
    }

    /**
     * Synchronize the Host and Device matrices and copies to host data to the device data.
     * This functions usually doesn't have to be called manually by the user. Instead, is called automatically
     * depending on which access methods are called by the user. This function also constructs the device
     * matrix if that has not been done yet. This allows the user to create host matrices and only create
     * the corresponding device matrix when it is needed.
    */
    CUDAMatrix<T>& hostToDeviceSync() {
        // If the matrix does not exist on device yet, create it from host parameters
        if ( not is_on_device and total_size > 0 )
            constructDevice( rows, cols, name );
        // Log this action
        if ( global_matrix_transfer_log )
            std::cout << EscapeSequence::GRAY << "Copying " << rows << "x" << cols << " matrix to device matrix '" << name << "'" << EscapeSequence::RESET << std::endl;
        // Because we use std::vectors for host and device when using gcc, and thrust::host_vector and thrust::device_vector when using nvcc, we
        // can just call device_data = host_data and std:: or thrust:: will take care of the rest. Internally, this will memcopy in both cases.
        device_data = host_data;
        // The Device Matrix is now ahead of the Host matrix
        host_is_ahead = false;
        // Return this pointer.
        return *this;
    }

    /**
     * Synchronize the Host and Device matrices and copies the device data to the host.
     * This functions usually doesn't have to be called manually by the user. Instead, is called automatically
     * depending on which access methods are called by the user. This function also constructs the device
     * matrix if that has not been done yet. This allows the user to create host matrices and only create
     * the corresponding device matrix when it is needed.
    */
    CUDAMatrix<T>& deviceToHostSync() {
        // If the matrix does not exist on host yet, create it from device parameters
        if ( not is_on_host and total_size > 0 )
            constructHost( rows, cols, name );
        // Log this data
        if ( global_matrix_transfer_log )
            std::cout << EscapeSequence::GRAY << "Copying " << rows << "x" << cols << " matrix from device matrix '" << name << "'" << EscapeSequence::RESET << std::endl;
        // Because we use std::vectors for host and device when using gcc, and thrust::host_vector and thrust::device_vector when using nvcc, we
        // can just call host_data = device_data and std:: or thrust:: will take care of the rest. Internally, this will memcopy in both cases.
        host_data = device_data;
        // The Host Matrix is now ahead of the device matrix
        host_is_ahead = true;
        // Return this pointer
        return *this;
    }

    /**
     * Swaps the pointers of the CUDAMatrix objects, as long as their sizes match.
     * This calls std::swap on the host and device vectors. Using nvcc or gcc, this will
     * simply switch the data pointers and not copy any data.
    */
    void swap( CUDAMatrix<T>& other ) {
        if ( rows != other.rows or cols != other.cols )
            return;
        std::swap( device_data, other.device_data );
        std::swap( host_data, other.host_data );
    }

    /**
     * Helper Enum for direction of slicing data. 
     * Right now, only column slicing is supported.
    */
    enum class Direction {
        ROW,
        COLUMN
    };

    /**
     * Slice the host data and return the slice
     * @param start: the starting point in the buffer. This is a scalar value and not a row/col.
     * This means slice() copies not a row or col, but starts somewhere in the matrix and then copies
     * [size] values.
     * @param size: the number of values to copy
     * @return: a host_vector of size [size]
    */
    Type::host_vector<T> slice( const int start, const int size, Direction direction = Direction::ROW ) {
        // Create a temporary buffer that will hold the slice
        Type::host_vector<T> buffer_out( size );
        // Because we use either a thrust::host_vector or a std::vector, we can just call std::copy on the iterators.
        std::copy( host_data.begin() + start, host_data.begin() + start + size, buffer_out.begin() );
        // And return the final buffer
        return buffer_out;
    }

    /**
     * Slice the device data and return the slice.
     * @param start: the starting point in the buffer. This is a scalar value and not a row/col.
     * This means slice() copies not a row or col, but starts somewhere in the matrix and then copies
     * [size] values.
     * @param size: the number of values to copy
     * @return: a host_vector of size [size]
    */
    std::vector<T> sliceDevice( const int start, const int size, Direction direction = Direction::ROW ) {
        // Create a temporary buffer that will hold the slice
        Type::host_vector<T> buffer_out( size );
        // In this case, we need to use std::copy for gcc and thrust::copy for nvcc.
        #ifdef USE_CPU
            std::copy( device_data.begin() + start, device_data.begin() + start + size, buffer_out.begin() );
        #else
            thrust::copy( device_data.begin() + start, device_data.begin() + start + size, buffer_out.begin() );
        #endif
        // And return the final buffer
        return buffer_out;
    }

    /**
     * Returns the total size of this matrix.
    */
    inline unsigned int getTotalSize() const {
        return total_size;
    }

    /**
     * Returns the raw pointer to the device memory. This is used in the Kernels, because they 
     * cannot directly work with std::vector or thrust::device_vectors.
    */
    inline T* getDevicePtr() {
        // If the host is ahead, synchronize first
        if ( host_is_ahead )
            hostToDeviceSync();
        // Return .data() if using gcc, and a raw_pointer_cast if using gcc.
        #ifdef USE_CPU
            return device_data.data();
        #else
            // Get the smart pointer from device_data.data() and convert it into a raw pointer
            return thrust::raw_pointer_cast( device_data.data() );
        #endif
    }

    /**
     * Returns the raw pointer to the host memory. 
    */
    inline T* getHostPtr() {
        // If the device is ahead, synchronize first
        if ( not host_is_ahead )
            deviceToHostSync();
        // In this case, we can just return .data() on either std::vector or thrust::host_vector
        return host_data.data();
    }

    /**
     * Returns the host vector. The return type is const because you are not supposed to manipulated
     * the data this way.
    */

   inline const Type::host_vector<T>& getHostVector() {
        if ( not host_is_ahead )
            deviceToHostSync();
        return host_data;
   }

    /**
     * .begin() and .end() iterators for the device and host data.
     * These can be used functions like thrust::reduce or std::reduce
    */
    inline auto dbegin() {
        return device_data.begin();
    }
    inline auto dend() {
        return device_data.end();
    }
    inline auto hbegin() {
        return host_data.begin();
    }
    inline auto hend() {
        return host_data.end();
    }

    /**
     * Row, col and index getters for host matrix 
     * The [] operator does synchronize the host and device memory
    */
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
        return host_data[index];
    }
    inline T operator()( int row, int column ) {
        return getHostPtr()[row * rows + column];
    }
    inline T deviceAt(int index) const {
        return device_data[index];
    }
};

} // namespace PC3