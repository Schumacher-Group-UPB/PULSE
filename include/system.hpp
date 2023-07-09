#pragma once
#include <complex>
using Scalar = std::complex<double>;
using namespace std::complex_literals;
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <vector>

/**
* @brief Very lightweight System Class containing all of the required system variables

*/
class System {
    public:
    // SI Rescaling Units
    double m_e = 9.10938356E-31;
    double h_bar = 1.0545718E-34;
    double e_e = 1.60217662E-19;
    double h_bar_s = 6.582119514E-4;

    // System Variables
    double m_eff;
    double gamma_c = 0.15;                // ps^-1
    double gamma_r = 1.5 * gamma_c;       // ps^-1
    double g_c = 3.E-6 / h_bar_s;         // meV mum^2
    double g_r = 2. * g_c;                // meV mum^2
    double R = 0.01;                      // ps^-1 mum^2
    //double P0 = 100.;                     // ps^-1 mum^-2
    //double w = 10.;                       // mum
    double xmax = 100.;                   // mum
    double g_pm = -g_c / 5;               // meV mum^2
    double delta_LT = 0.025E-3 / h_bar_s; // meV
    double m_plus = 0.;
    double m_minus = 0.;

    // Numerics
    int s_N = 400;
    double dx;
    double dt;
    double t;
    double t_max = 1000;
    bool normalize_phase_states = true;
    bool normalizePhasePulse = false;
    double fft_every = 1; // ps
    int iteration = 0;

    // Pump arrays
    std::vector<double> pump_amp;
    std::vector<double> pump_width;
    std::vector<double> pump_X;
    std::vector<double> pump_Y;
    std::vector<int> pump_pol;

    // Pulse arrays
    std::vector<double> pulse_t0;
    std::vector<double> pulse_amp;
    std::vector<double> pulse_freq;
    std::vector<double> pulse_sigma;
    std::vector<int> pulse_m;
    std::vector<int> pulse_pol;
    std::vector<double> pulse_width;
    std::vector<double> pulse_X;
    std::vector<double> pulse_Y;
};

class Buffer {
    public:
    Scalar* u_P_plus;
    Scalar* u_P_minus;
    Scalar* Psi_Plus;
    Scalar* Psi_Minus;
    Scalar* n_Plus;
    Scalar* n_Minus;
    Scalar* fft_plus;
    Scalar* fft_minus;

    Buffer(const int N) {
        u_P_plus = new Scalar[N * N];
        u_P_minus = new Scalar[N * N];
        Psi_Plus = new Scalar[N * N];
        Psi_Minus = new Scalar[N * N];
        n_Plus = new Scalar[N * N];
        n_Minus = new Scalar[N * N];
        fft_plus = new Scalar[N * N];
        fft_minus = new Scalar[N * N];
    }
    Buffer(const System& s) : Buffer(s.s_N) {}
    Buffer() {};
};

class MatrixHandler {
    public:
    std::map<std::string, std::ofstream> files;
    std::string outputPath = "data";
    std::string loadPath = "";
    std::string outputName = "";
    std::string colorPalette = "resources/colormap.txt";
    int plotmodulo = 5;
    bool disableRender = false;

    MatrixHandler() {};
    MatrixHandler(MatrixHandler& other) : files(std::move(other.files)), outputPath(other.outputPath), 
    outputName(other.outputName), colorPalette(other.colorPalette), plotmodulo(other.plotmodulo), loadPath(other.loadPath),
    disableRender(other.disableRender) {};

    std::string toPath(const std::string& name) {
        return outputPath + (outputPath.back() == '/' ? "" : "/") + outputName + (outputName.empty() ? "" : "_") + name + ".txt";
    }

    std::ofstream& getFile(const std::string& name) {
        if (files.find(name) == files.end())
            files[name] = std::ofstream(toPath(name));
        return files[name];
    }

    void loadMatrixFromFile(const std::string& filepath, Scalar* buffer) {
        std::ifstream filein; 
        filein.open(filepath, std::ios::in);
        std::istringstream inputstring; 
        std::string line;
        int i = 0;
        double x,y,re,im;
        if (filein.is_open()) {
            while(getline(filein,line)) {
                if (line.size()>2) {
                    inputstring = std::istringstream(line);
                    inputstring >> x >> y >> re >> im;
                    buffer[i] = re + 1.0i*im;
                    i++;
                }
            }
            filein.close();
            std::cout << "Loaded " << i << " elements from " << filepath << std::endl;
        } else {
            std::cout << "Error: Couldn't load " << filepath << std::endl;
        }
    }

    void outputMatrixToFile( const Scalar* buffer, int row_start, int row_stop, int col_start, int col_stop, const System& s, std::ofstream& out, const std::string& name ) {
        for ( int i = row_start; i < row_stop; i++ ) {
            for ( int j = col_start; j < col_stop; j++ ) {
                auto index = j + i * s.s_N;
                auto x = -s.xmax / 2 + j * s.dx;
                auto y = -s.xmax / 2 + i * s.dx;
                out << x << " " << y << " " << std::setprecision(10) << std::real( buffer[index] ) << " " << std::imag( buffer[index] ) << std::endl;
            }
            out << std::endl;
        }
        std::cout << "Output " << (row_stop-row_start)*(col_stop-col_start) << " elements to " <<  toPath(name) << "." << std::endl;
    }
    void outputMatrixToFile( const Scalar* buffer, int row_start, int row_stop, int col_start, int col_stop, const System& s, const std::string& out ) {
        auto& file = getFile(out);
        outputMatrixToFile( buffer, row_start, row_stop, col_start, col_stop, s, file, out );
    }
    void outputMatrixToFile( const Scalar* buffer, const System& s, const std::string& out ) {
        auto& file = getFile(out);
        outputMatrixToFile( buffer, 0, s.s_N, 0, s.s_N, s, file, out );
    }
    void outputMatrixToFile( const Scalar* buffer, const System& s, std::ofstream& out, const std::string& name ) {
        outputMatrixToFile( buffer, 0, s.s_N, 0, s.s_N, s, out, name );
    }

    void outputMatrices( System& system, Buffer& buffer ) {
    std::vector<std::string> fileoutputkeys = { "Psi_plus", "Psi_minus", "n_plus", "n_minus", "fft_plus", "fft_minus" };
#pragma omp parallel for
    for ( int i = 0; i < fileoutputkeys.size(); i++ ) {
        auto key = fileoutputkeys[i];
        if ( key == "Psi_plus" )
            outputMatrixToFile( buffer.Psi_Plus, system, key );
        if ( key == "Psi_minus" )
            outputMatrixToFile( buffer.Psi_Minus, system, key );
        if ( key == "n_plus" )
            outputMatrixToFile( buffer.n_Plus, system, key );
        if ( key == "n_minus" )
            outputMatrixToFile( buffer.n_Minus, system, key );
        if ( key == "fft_plus" )
            outputMatrixToFile( buffer.fft_plus, system, key );
        if ( key == "fft_minus" )
            outputMatrixToFile( buffer.fft_minus, system, key );
    }
}

void loadMatrices( System& system, Buffer& buffer ) {
    if (loadPath.size() < 1) 
        return;
    std::cout << "Loading Matrices from " << loadPath << std::endl;
    if ( loadPath.back() != '/' )
        loadPath += "/";
    std::vector<std::string> fileoutputkeys = { "Psi_plus", "Psi_minus", "n_plus", "n_minus" };
#pragma omp parallel for
    for ( auto i = 0; i < fileoutputkeys.size(); i++) {
        if ( fileoutputkeys[i] == "Psi_plus" )
            loadMatrixFromFile( loadPath + fileoutputkeys[i] + ".txt", buffer.Psi_Plus );
        else if ( fileoutputkeys[i] == "Psi_minus" )
            loadMatrixFromFile( loadPath + fileoutputkeys[i] + ".txt", buffer.Psi_Minus );
        else if ( fileoutputkeys[i] == "n_plus" )
            loadMatrixFromFile( loadPath + fileoutputkeys[i] + ".txt", buffer.n_Plus );
        else if ( fileoutputkeys[i] == "n_minus" )
            loadMatrixFromFile( loadPath + fileoutputkeys[i] + ".txt", buffer.n_Minus );
    }
}
};