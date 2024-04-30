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

    struct Header {
        // Spatial Parameters
        real_number L_x, L_y;
        real_number dx, dy;
        // Time Parameter
        real_number t;
        // Oscillator Parameters
        real_number t0,freq,sigma;

        Header() : L_x( 0 ), L_y( 0 ), dx( 0 ), dy( 0 ), t( 0 ), t0(0), freq(0), sigma(0) {}
        Header( real_number L_x, real_number L_y, real_number dx, real_number dy, real_number t ) : Header() {
            this->L_x = L_x;
            this->L_y = L_y;
            this->dx = dx;
            this->dy = dy;
            this->t = t;
            this->t0 = 0;
            this->freq = 0;
            this->sigma = 0;
        }
        Header( real_number L_x, real_number L_y, real_number dx, real_number dy, real_number t, real_number t0, real_number freq, real_number sigma ) : Header() {
            this->L_x = L_x;
            this->L_y = L_y;
            this->dx = dx;
            this->dy = dy;
            this->t = t;
            this->t0 = t0;
            this->freq = freq;
            this->sigma = sigma;
        }

        friend std::ostream& operator<<( std::ostream& os, const Header& header ) {
            os << "LX " << header.L_x << " LY " << header.L_y << " DX " << header.dx << " DY " << header.dy << " TIME " << header.t;
            if (header.t0 != 0 and header.freq != 0 and header.sigma != 0)
                os << " OSC T0 " << header.t0 << " FREQ " << header.freq << " SIGMA " << header.sigma;
            return os;
        }
    };

    std::string toPath( const std::string& name );

    std::ofstream& getFile( const std::string& name );

    bool loadMatrixFromFile( const std::string& filepath, complex_number* buffer );
    bool loadMatrixFromFile( const std::string& filepath, real_number* buffer );

    void outputMatrixToFile( const complex_number* buffer, unsigned int col_start, unsigned int col_stop, unsigned int row_start, unsigned int row_stop, const unsigned int N_x, const unsigned int N_y, unsigned int increment, const Header& header, std::ofstream& out, const std::string& name );
    void outputMatrixToFile( const complex_number* buffer, unsigned int col_start, unsigned int col_stop, unsigned int row_start, unsigned int row_stop, const unsigned int N_x, const unsigned int N_y, unsigned int increment, const Header& header, const std::string& out );
    void outputMatrixToFile( const complex_number* buffer, const unsigned int N_x, const unsigned int N_y, const Header& header, const std::string& out );
    void outputMatrixToFile( const complex_number* buffer, const unsigned int N_x, const unsigned int N_y, const Header& header, std::ofstream& out, const std::string& name );

    void outputMatrixToFile( const real_number* buffer, unsigned int col_start, unsigned int col_stop, unsigned int row_start, unsigned int row_stop, const unsigned int N_x, const unsigned int N_y, unsigned int increment, const Header& header, std::ofstream& out, const std::string& name );
    void outputMatrixToFile( const real_number* buffer, unsigned int col_start, unsigned int col_stop, unsigned int row_start, unsigned int row_stop, const unsigned int N_x, const unsigned int N_y, unsigned int increment, const Header& header, const std::string& out );
    void outputMatrixToFile( const real_number* buffer, const unsigned int N_x, const unsigned int N_y, const Header& header, const std::string& out );
    void outputMatrixToFile( const real_number* buffer, const unsigned int N_x, const unsigned int N_y, const Header& header, std::ofstream& out, const std::string& name );

    void init( int argc, char** argv );
};

std::vector<char*> readConfigFromFile(int argc, char** argv);

} // namespace PC3