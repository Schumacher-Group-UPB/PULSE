#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
// ofstream
#include <fstream>
#include "cuda/typedef.cuh"
#include "cuda/cuda_macro.cuh"
#include "kernel/kernel_halo.cuh"
#include "misc/escape_sequences.hpp"
#include "cuda/cuda_matrix_base.hpp"
#include "misc/commandline_io.hpp"

namespace PC3 {

/**
 * @brief CUDA Wrapper for a matrix
 * Handles memory management and host-device synchronization when
 * using the getDevicePtr() and getHostPtr() functions.
 * 
 * NEW: The main purpose of this class remains the communication between
 * the full size host matrices (NxN) and the device matrices, which are halo'ed
 * and may be composed of multiple MxM subgrids, each with its own device pointer.
 *
 * @tparam T either Type::real or Type::complex
 */
template <typename T>
class CUDAMatrix : CUDAMatrixBase {
   private:
    // Rows and Cols of the Matrix. These can be different, so rows =/= cols
    Type::uint rows, cols;
    Type::uint subgrid_rows, subgrid_cols, subgrid_rows_with_halo, subgrid_cols_with_halo;
    // The total Size of the Matrix = rows*cols. Used for allocation purposes.
    Type::uint total_size_host;
    Type::uint total_size_device;
    // The number of subgrids the device matrix is composed of
    Type::uint subgrids_per_dim;
    Type::uint total_num_subgrids; // = subgrids_per_dim * subgrids_per_dim;
    Type::uint subgrid_size; // = subgrid_rows * subgrid_cols;
    Type::uint subgrid_size_with_halo; // = subgrid_rows_with_halo * subgrid_cols_with_halo;
    // The size of the halo around each subgrid
    Type::uint halo_size;
    // Name of the Matrix. Mostly used for debugging purposes.
    std::string name;

    // Device Vector. When using nvcc, this is a thrust::device_vector. When using gcc, this is a std::vector
    Type::host_vector<Type::device_vector<T>> device_data;
    Type::device_vector<T*> subgrid_pointers_device;
    // Host Vector. When using nvcc, this is a thrust::host_vector. When using gcc, this is a std::vector
    Type::host_vector<T> host_data;

    // Static buffer for the full size device matrix. This is used when the full grid is required on the device, for example
    // when doing a FFT or when synchronizing the device with the host matrices. We specifically use a map here to allow
    // for multiple sizes of matrices.
    inline static std::map<Type::uint,Type::device_vector<T>> device_data_full;

    // Size of this Matrix. Mostly used for debugging and the Summary    
    double size_in_mb_host;
    double size_in_mb_device;

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
        total_size_host = 0;
        total_size_device = 0;
        is_on_device = false;
        is_on_host = false;
        host_is_ahead = false;
    };

    CUDAMatrix( CUDAMatrix& other ) : rows( other.rows ), cols( other.cols ), total_size_host( other.total_size_host ), total_size_device( other.total_size_device ), name( other.name ), device_data( other.device_data ), host_data( other.host_data ) {
        other.total_size_host = 0;
        other.total_size_device = 0;
        other.cols = 0;
        other.rows = 0;
    }
    CUDAMatrix( CUDAMatrix&& other ) : rows( other.rows ), cols( other.cols ), total_size_host( other.total_size_host ), total_size_device( other.total_size_device ), name( other.name ), device_data( other.device_data ), host_data( other.host_data ) {
        other.total_size_host = 0;
        other.total_size_device = 0;
        other.cols = 0;
        other.rows = 0;
    }

    CUDAMatrix( Type::uint rows, Type::uint cols, const std::string& name ) : rows( rows ), cols( cols ), name( name ) {
        construct( rows, cols, name );
    }

    CUDAMatrix( Type::uint rows, Type::uint cols, T* data, const std::string& name ) : CUDAMatrix( rows, cols, name ) {
        setTo( data );
    }

    CUDAMatrix( Type::uint root_size, const std::string& name ) : CUDAMatrix( root_size, root_size, name ){};
    CUDAMatrix( Type::uint root_size, T* data, const std::string& name ) : CUDAMatrix( root_size, root_size, data, name ){};

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
        if ( not isOnHost() ) {
            std::cout << PC3::CLIO::prettyPrint( "Matrix '" + name + "' is not on host!", PC3::CLIO::Control::FullError) << std::endl;
            return *this;
        }
        // Set the host data to the current data. This works fine with both std::vector and thrust::vector
        host_data = data;
        // The host data has been updated, so the host is now ahead.
        host_is_ahead = true;
        // Log this action
        if (global_matrix_transfer_log)
            std::cout << PC3::CLIO::prettyPrint( "Copied " + std::to_string(data.size()) + " elements to '" + getName() + "'", PC3::CLIO::Control::Info | PC3::CLIO::Control::Secondary ) << std::endl; 
        // Return this pointer
        return *this;
    }
    
    /**
     * Sets the host data of this matrix to the other matrix' data
     * @param other: Other CUDAMatrix of same size
    */
    CUDAMatrix<T>& setTo( CUDAMatrix<T>& other ) {
        // If this matrix is not on host, we cannot set its host data. 
        if ( not isOnHost() or not other.isOnHost()) {
            std::cout << PC3::CLIO::prettyPrint( "Matrix '" + name + "' or Matrix '" + other.getName() + "' is not on host.", PC3::CLIO::Control::FullError) << std::endl;
            return *this;
        }
        // Set the host data to the current data. This works fine with both std::vector and thrust::vector
        host_data = other.getHostVector();
        // The host data has been updated, so the host is now ahead.
        host_is_ahead = true;
        // Log this action
        if (global_matrix_transfer_log)
            std::cout << PC3::CLIO::prettyPrint( "Copied '" + other.getName() + "' to '" + getName() + "'", PC3::CLIO::Control::Info | PC3::CLIO::Control::Secondary ) << std::endl; 
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
        if ( host_data.empty() or rows == 0 or cols == 0 or total_size_host == 0 )
            return *this;
        // Subtract the current size of this host matrix from the global matrix size
        global_total_host_mb -= size_in_mb_host;
        // Log this action. Mostly for simple debugging.
        if ( global_matrix_creation_log )
            std::cout << PC3::CLIO::prettyPrint( "Freeing " + std::to_string(rows) + "x" + std::to_string(rows) + " matrix '" + name + "' from host, total allocated host space: " + std::to_string(global_total_host_mb) + " MB.", PC3::CLIO::Control::Info | PC3::CLIO::Control::Secondary) << std::endl;
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
        if ( device_data.empty() or rows == 0 or cols == 0 or total_size_device == 0 )
            return *this;
        // Subtract the current size of this device matrix from the global matrix size
        global_total_device_mb -= size_in_mb_device;
        // Log this action. Mostly for simple debugging.
        if ( global_matrix_creation_log )
            std::cout << PC3::CLIO::prettyPrint( "Freeing " + std::to_string(rows) + "x" + std::to_string(cols) + " matrix '" + name + "' from device, total allocated device space: " + std::to_string(global_total_device_mb) + " MB." , PC3::CLIO::Control::Info | PC3::CLIO::Control::Secondary) << std::endl;
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

    void calculateSizes() {
        total_size_host = rows * cols;
        size_in_mb_host = total_size_host * sizeof( T ) / 1024.0 / 1024.0;
        total_size_device = (rows+2*halo_size) * (cols+2*halo_size);
        size_in_mb_device = total_size_device * sizeof( T ) / 1024.0 / 1024.0;
        subgrid_rows = rows / subgrids_per_dim;
        subgrid_rows_with_halo = subgrid_rows + 2*halo_size;
        subgrid_cols = cols / subgrids_per_dim;
        subgrid_cols_with_halo = subgrid_cols + 2*halo_size;
        total_num_subgrids = subgrids_per_dim * subgrids_per_dim;
        subgrid_size = subgrid_rows * subgrid_cols;
        subgrid_size_with_halo = subgrid_rows_with_halo * subgrid_cols_with_halo;
    }

    /**
     * Constructs the Device Matrix vector. This function only allocates the memory and does not copy any data.
     * @return: ptr to this matrix.
    */
    CUDAMatrix<T>& constructDevice( Type::uint rows, Type::uint cols, Type::uint subgrids_per_dim, Type::uint halo_size, const std::string& name ) {
        this->name = name;
        this->rows = rows;
        this->cols = cols;
        this->subgrids_per_dim = subgrids_per_dim;
        this->halo_size = halo_size;
        // Calculate the total size of this matrix as well as its size in bytes
        calculateSizes();
        if (this->rows * this->cols == 0) {
            return *this;
        }
        // Add the size to the global counter for the device sizes and update the maximum encountered memory size
        global_total_device_mb += size_in_mb_device;
        global_total_device_mb_max = std::max( global_total_device_mb, global_total_device_mb_max );
        // Log this action.
        if ( global_matrix_creation_log )
            std::cout << PC3::CLIO::prettyPrint( "Allocating " + std::to_string(size_in_mb_device) + " MB for " + std::to_string(rows) + "x" + std::to_string(cols) + " device matrix '" + name + "' with halo " + std::to_string(halo_size) + " and " + std::to_string(total_num_subgrids) + " total subgrids, total allocated device space: " + std::to_string(global_total_device_mb) + " MB." , PC3::CLIO::Control::Info | PC3::CLIO::Control::Secondary) << std::endl;
        // Reserve space on the device. When using nvcc, this allocates device memory on the GPU using thrust. When using gcc, this allocates memory for the CPU using std::vector.
        // We use the [..] operator for device arrays provided by thrust, which is slow but we only do it when initializing the matrix.
        subgrid_pointers_device = Type::device_vector<T*>( total_num_subgrids );
        for (int i = 0; i < total_num_subgrids; i++) {
            device_data.push_back( Type::device_vector<T>( total_size_device, (T)0.0 ) );        
            subgrid_pointers_device[i] = GET_RAW_PTR( device_data.back() );
        }
        if (not device_data_full.count(total_size_host)) {
            if (global_matrix_creation_log)
                std::cout << PC3::CLIO::prettyPrint( "Full grid buffer for size " + std::to_string(total_size_host) + " not found, creating new buffer.", PC3::CLIO::Control::Info | PC3::CLIO::Control::Secondary) << std::endl;
            device_data_full[total_size_host] = Type::device_vector<T>( total_size_host );
        }
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
    CUDAMatrix<T>& constructDevice( Type::uint root_size, const std::string& name ) {
        return constructDevice( root_size, root_size, 1, 0, name );
    }

    /**
     * Constructs the Host Matrix vector. This function only allocates the memory and does not copy any data.
     * @return: ptr to this matrix.
    */
    CUDAMatrix<T>& constructHost( Type::uint rows, Type::uint cols, Type::uint subgrids_per_dim, Type::uint halo_size, const std::string& name = "unnamed" ) {
        this->name = name;
        this->rows = rows;
        this->cols = cols;
        this->subgrids_per_dim = subgrids_per_dim;
        this->halo_size = halo_size;
        // Calculate the total size of this matrix as well as its size in bytes
        calculateSizes();
        // Calculate the total size of this matrix as well as its size in bytes
        size_in_mb_host = total_size_host * sizeof( T ) / 1024.0 / 1024.0;
        // Add the size to the global counter for the host sizes and update the maximum encountered memory size
        global_total_host_mb += size_in_mb_host;
        global_total_host_mb_max = std::max( global_total_host_mb, global_total_host_mb_max );
        // Log this action.
        if ( global_matrix_creation_log )
            std::cout << PC3::CLIO::prettyPrint( "Allocating " + std::to_string(size_in_mb_host) + " MB for " + std::to_string(rows) + "x" + std::to_string(cols) + " host matrix '" + name + "', total allocated host space: " + std::to_string(global_total_host_mb) + " MB." , PC3::CLIO::Control::Info | PC3::CLIO::Control::Secondary) << std::endl;
        // Reserve space on the device. When using nvcc, this allocates device memory on the GPU using thrust. When using gcc, this allocates memory for the CPU using std::vector.
        host_data.resize( total_size_host );
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
    CUDAMatrix<T>& constructHost( Type::uint root_size, const std::string& name = "unnamed" ) {
        return constructHost( root_size, root_size, 1, 0, name );
    }

    /**
     * Constructs both the Host and the Device Matrices.
     * This overwrites the name and the sizes, but we dont care.
     * This function also assumes no subgrids and no halo size.
    */
    CUDAMatrix<T>& construct( Type::uint rows, Type::uint cols, Type::uint subgrids_per_dim, Type::uint halo_size, const std::string& name = "unnamed" ) {
        constructDevice( rows, cols, subgrids_per_dim, halo_size, name );
        constructHost( rows, cols, subgrids_per_dim, halo_size, name );
        return *this;
    }

    /**
     * Alias for construct( rows, cols, name ) using root_size = rows*cols
    */
    CUDAMatrix<T>& construct( Type::uint root_size, const std::string& name = "unnamed" ) {
        return construct( root_size, root_size, 1, 0, name );
    }

    /**
     * Fill Matrix with set value. From host-side only.
     * If the matrix is also on the device, synchronize the data.
     */
    CUDAMatrix<T>& fill( T value ) {
        if ( not is_on_host )
            return *this;
        std::fill( host_data.begin(), host_data.end(), value );
        if ( is_on_device )
            hostToDeviceSync();
        return *this;
    }

    /**
     * Synchronize the Host and Device matrices and copies to host data to the device data.
     * This functions usually doesn't have to be called manually by the user. Instead, is called automatically
     * depending on which access methods are called by the user. This function also constructs the device
     * matrix if that has not been done yet. This allows the user to create host matrices and only create
     * the corresponding device matrix when it is needed.
     * NEW: This function now also handles to synchronization between subgrids on the device and the full size host matrix.
    */
    CUDAMatrix<T>& hostToDeviceSync() {
        // If the matrix is not on host and thus, has to be empty, do nothing
        if (not is_on_host or total_size_host == 0)
            return *this;
        
        // If the matrix does not exist on device yet, create it from host parameters
        if ( not is_on_device and subgrid_rows*subgrid_cols > 0 )
            constructDevice( rows, cols, subgrids_per_dim, halo_size, name );
        // Log this action
        if ( global_matrix_transfer_log )
            std::cout << PC3::CLIO::prettyPrint( "Copying " + std::to_string(rows) + "x" + std::to_string(cols) + " matrix to device matrix '" + name + "' from host." , PC3::CLIO::Control::Info | PC3::CLIO::Control::Secondary) << std::endl;

        // If the subgrid size is 1 and the halo_size is zero, we can just copy the full matrix to the device data
        if (subgrid_size == 1 and halo_size == 0) {
            device_data[0] = host_data;
        } else {
        // Otherwise, we have to copy the full matrix to the device data_full and then split the full matrix into subgrids
            device_data_full[total_size_host] = host_data;
            toSubgrids();
        }
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
        // If the matrix is not on device and thus, has to be empty, do nothing
        if (not is_on_device or total_size_device == 0)
            return *this;
        // If the matrix does not exist on host yet, create it from device parameters
        if ( not is_on_host and subgrid_rows*subgrid_cols > 0 )
            constructHost( rows, cols, subgrids_per_dim, halo_size, name );
        // Log this data
        if ( global_matrix_transfer_log )
            std::cout << PC3::CLIO::prettyPrint( "Copying " + std::to_string(rows) + "x" + std::to_string(cols) + " matrix from device matrix '" + name + "' to host.", PC3::CLIO::Control::Info | PC3::CLIO::Control::Secondary) << std::endl;

        if (subgrid_size == 1 and halo_size == 0) {
            host_data = device_data[0];
        } else {
            toFull();
            host_data = device_data_full[total_size_host];
        }
        // The Host Matrix is now ahead of the device matrix
        host_is_ahead = true;
        // Return this pointer
        return *this;
    }

    // DEbug purposes only
    void print() {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < subgrid_cols; j++) {
                std::cout << device_data[0][i*subgrid_cols+j] << " ";
            }
            std::cout << std::endl;
        }
    }
    // output device_vector[0] to file
    void tofile(std::string path) {
        std::ofstream file;
        file.open(path);
        for (int i = 0; i < subgrid_rows_with_halo; i++) {
            for (int j = 0; j < subgrid_rows_with_halo; j++) {
                T v = device_data[0][i*subgrid_cols_with_halo+j];
                file << CUDA::real(v) << " ";
            }
            file << std::endl;
        }
        file.close();
    }

    /**
     * Swaps the pointers of the CUDAMatrix objects, as long as their sizes match.
     * This calls std::swap on the host and device vectors. Using nvcc or gcc, this will
     * simply switch the data pointers and not copy any data.
       DEPRECATED; TODO: REMOVE
    */
    void swap( CUDAMatrix<T>& other ) {
        if ( rows != other.rows or cols != other.cols )
            return;
        std::swap( device_data, other.device_data );
        std::swap( host_data, other.host_data );
    }


    /**
     * Returns the total size of this matrix.
    */
    inline Type::uint getTotalSize() const {
        return total_size_host;
    }

    /**
     * Returns the raw pointer to the device memory. This is used in the Kernels, because they 
     * cannot directly work with std::vector or thrust::device_vectors.
    */
    inline T* getDevicePtr(Type::uint subgrid = 0) {
        // If the host is ahead, synchronize first
        if ( host_is_ahead )
            hostToDeviceSync();
        if (not is_on_device)
            return nullptr;
        return GET_RAW_PTR( device_data[subgrid] );
    }

    /**
     * Returns a vector of device pointers
     * @return T** - Pointer to the device pointers
    */
    inline T** getSubgridDevicePtrs() {
        return GET_RAW_PTR( subgrid_pointers_device );
    }

    /**
     * Returns the raw pointer to the host memory. 
     TODO: in zwei funktionen ändern:
     getHostPtr und getHostPtrSync, ersteres macht zwar deviceToHost aber setzt host_is_ahead = false damit 
     man nicht unnötig kopiert wenn man nur auslesen will.
     ansonsten: getSynchronizedHostPointer() und getReadHostPointer() 
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

    CUDAMatrix<T>& toFull( PC3::Type::stream_t stream = 0 ) {
        // Copy all device data to the full static device_data_full buffer
        // This function will copy the data within the respective halos, if they exist,
        // to the full matrix. 
        dim3 block_size( 256, 1 );
        dim3 grid_size( ( rows*cols + block_size.x ) / block_size.x, 1 );
        auto fullgrid_dev_ptr = GET_RAW_PTR(device_data_full[total_size_host]);
        auto dev_ptrs = getSubgridDevicePtrs();
        if (global_matrix_transfer_log)
            std::cout << PC3::CLIO::prettyPrint( "Copying " + std::to_string(subgrids_per_dim) + "x" + std::to_string(subgrids_per_dim) + " subgrids to full grid buffer for matrix '" + name + "'", PC3::CLIO::Control::Info | PC3::CLIO::Control::Secondary) << std::endl;
        CALL_KERNEL(PC3::Kernel::Halo::halo_grid_to_full_grid, "halo_to_full:"+name, grid_size, block_size, stream, cols, rows, subgrids_per_dim, subgrid_cols, subgrid_rows, halo_size, fullgrid_dev_ptr, dev_ptrs);
        return *this;
    }

    CUDAMatrix<T>& toSubgrids( PC3::Type::stream_t stream = 0 ) {
        // Copy all device data of the full static device_data_full buffer
        // This function will copy the data within the full matrix to the respective subgrid
        // The halos will remain unsynchronized
        dim3 block_size( 256, 1 );
        dim3 grid_size( ( rows*cols + block_size.x ) / block_size.x, 1 );
        auto fullgrid_dev_ptr = fullMatrixPointer();
        auto dev_ptrs = getSubgridDevicePtrs();
        if (global_matrix_transfer_log)
            std::cout << PC3::CLIO::prettyPrint( "Copying full grid buffer to subgrids for matrix '" + name + "'", PC3::CLIO::Control::Info | PC3::CLIO::Control::Secondary) << std::endl;
        CALL_KERNEL(PC3::Kernel::Halo::full_grid_to_halo_grid, "full_to_halo:"+name, grid_size, block_size, stream, cols, rows, subgrids_per_dim, subgrid_cols, subgrid_rows, halo_size, fullgrid_dev_ptr, dev_ptrs);
        return *this;
    }

    // Only call these after toFull()
    T* fullMatrixPointer() {
        return GET_RAW_PTR(device_data_full[total_size_host]);
    }
    
    auto fullMatrixBegin() {
        return device_data_full[total_size_host].begin();
    }
    auto fullMatrixEnd() {
        return device_data_full[total_size_host].end();
    }

    // TODO: wie genau macht man das jetzt mit der subgrid synchro? kernel um einzelne subgrids in großes gridbuffer zu kopieren?
    // dann hier: dbegin() -> copy to static buffer -> return buffer adress

    /**
     * .begin() and .end() iterators for the device and host data.
     * These can be used functions like thrust::reduce or std::reduce
    */
    inline auto dbegin() {
        return fullMatrixBegin();
    }
    inline auto dend() {
        return fullMatrixEnd();
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
    inline T deviceAt(int index, const Type::uint subgrid = 0) const {
        return device_data[subgrid][index];
    }
};

} // namespace PC3