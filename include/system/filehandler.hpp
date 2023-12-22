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

    bool disableRender;

    FileHandler();
    FileHandler( int argc, char** argv);
    FileHandler( FileHandler& other ) = delete;

    std::string toPath( const std::string& name );

    std::ofstream& getFile( const std::string& name );

    void loadMatrixFromFile( const std::string& filepath, complex_number* buffer );
    void loadMatrixFromFile( const std::string& filepath, real_number* buffer );

    void outputMatrixToFile( const complex_number* buffer, int col_start, int col_stop, int row_start, int row_stop, const unsigned int N_x, const unsigned int N_y, unsigned int increment, const real_number s_L_x, const real_number s_L_y, const real_number dx, const real_number dy, std::ofstream& out, const std::string& name );
    void outputMatrixToFile( const complex_number* buffer, int col_start, int col_stop, int row_start, int row_stop, const unsigned int N_x, const unsigned int N_y, unsigned int increment, const real_number s_L_x, const real_number s_L_y, const real_number dx, const real_number dy, const std::string& out );
    void outputMatrixToFile( const complex_number* buffer, const unsigned int N_x, const unsigned int N_y, const real_number s_L_x, const real_number s_L_y, const real_number dx, const real_number dy, const std::string& out );
    void outputMatrixToFile( const complex_number* buffer, const unsigned int N_x, const unsigned int N_y, const real_number s_L_x, const real_number s_L_y, const real_number dx, const real_number dy, std::ofstream& out, const std::string& name );

    void outputMatrixToFile( const real_number* buffer, int col_start, int col_stop, int row_start, int row_stop, const unsigned int N_x, const unsigned int N_y, unsigned int increment, const real_number s_L_x, const real_number s_L_y, const real_number dx, const real_number dy, std::ofstream& out, const std::string& name );
    void outputMatrixToFile( const real_number* buffer, int col_start, int col_stop, int row_start, int row_stop, const unsigned int N_x, const unsigned int N_y, unsigned int increment, const real_number s_L_x, const real_number s_L_y, const real_number dx, const real_number dy, const std::string& out );
    void outputMatrixToFile( const real_number* buffer, const unsigned int N_x, const unsigned int N_y, const real_number s_L_x, const real_number s_L_y, const real_number dx, const real_number dy, const std::string& out );
    void outputMatrixToFile( const real_number* buffer, const unsigned int N_x, const unsigned int N_y, const real_number s_L_x, const real_number s_L_y, const real_number dx, const real_number dy, std::ofstream& out, const std::string& name );
    
    void init( int argc, char** argv );
};

} // namespace PC3