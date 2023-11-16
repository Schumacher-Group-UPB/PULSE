#pragma once

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include "cuda/cuda_complex.cuh"

namespace PC3 {

class FileHandler {
   public:
    std::map<std::string, std::ofstream> files;
    std::string outputPath, loadPath, outputName, color_palette, color_palette_phase;

    int out_modulo;
    bool disableRender;

    FileHandler();
    FileHandler( int argc, char** argv);
    FileHandler( FileHandler& other ) = delete;

    std::string toPath( const std::string& name );

    std::ofstream& getFile( const std::string& name );

    void loadMatrixFromFile( const std::string& filepath, complex_number* buffer );
    void loadMatrixFromFile( const std::string& filepath, real_number* buffer );

    void outputMatrixToFile( const complex_number* buffer, int row_start, int row_stop, int col_start, int col_stop, const unsigned int N, const real_number xmax, const real_number dx, std::ofstream& out, const std::string& name );
    void outputMatrixToFile( const complex_number* buffer, int row_start, int row_stop, int col_start, int col_stop, const unsigned int N, const real_number xmax, const real_number dx, const std::string& out );
    void outputMatrixToFile( const complex_number* buffer, const unsigned int N, const real_number xmax, const real_number dx, const std::string& out );
    void outputMatrixToFile( const complex_number* buffer, const unsigned int N, const real_number xmax, const real_number dx, std::ofstream& out, const std::string& name );

    void outputMatrixToFile( const real_number* buffer, int row_start, int row_stop, int col_start, int col_stop, const unsigned int N, const real_number xmax, const real_number dx, std::ofstream& out, const std::string& name );
    void outputMatrixToFile( const real_number* buffer, int row_start, int row_stop, int col_start, int col_stop, const unsigned int N, const real_number xmax, const real_number dx, const std::string& out );
    void outputMatrixToFile( const real_number* buffer, const unsigned int N, const real_number xmax, const real_number dx, const std::string& out );
    void outputMatrixToFile( const real_number* buffer, const unsigned int N, const real_number xmax, const real_number dx, std::ofstream& out, const std::string& name );

    void init( int argc, char** argv );
};

} // namespace PC3