#pragma once
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdio.h> // remove files
#include <math.h>
#include "nvrtc_wrap.h"
#include "bunch.h"
#include "timer.h"
#include "utils.h"

/*//////////////////
KERNEL FUNCTION EXAMPLE

std::string kernel = "                                          \n\
#include \"elements.h\"                                         \n\
extern \"C\" __global__                                         \n\
void track(size_t n, Particle* p)                               \n\
{                                                               \n\
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;           \n\
  if (tid < n) {                                                \n\
    Drift d(0.77)(p[tid]);                                      \n\
    Quad qf(0.09)(p[tid]);                                      \n\
    Quad qd(-0.09)(p[tid]);                                     \n\
    Multipole sx(2,1e-4)(p[tid]);                               \n\
  }                                                             \n\
}                                                               \n";

*///////////////////

class Lattice {
  std::vector<std::string> lattice;
  bool edited_lattice = true; //flag indicating if recompilation is required

  std::vector<NVRTC> nvrtcs; //modules for kernels managment

 public:
  size_t n_turns = 1;
  // Initial conditions, populated manually or when reading the mad twiss table
  double BETX = 0.;
  double BETY = 0.;
  double ALFX = 0.;
  double ALFY = 0.;
  double DX   = 0.;
  double DY   = 0.;
  double DPX  = 0.;
  double DPY  = 0.;
  double X    = 0.; //closed orbit
  double Y    = 0.;
  double PX   = 0.;
  double PY   = 0.;

  // Some parameters used together with the above ones to create matched bunches
  double norm_emit_x = 0.;
  double norm_emit_y = 0.;
  double bunch_length = 0.;
  double bunch_energy_spread = 0.;
  double energy = 0.;
  double mass   = 0.93827203; //Proton mass in GeV
  // Turn by turn data is no more than copies of the bunch
  std::vector<HostBunch> turn_by_turn_data;
  size_t collect_tbt_data = 0; //indicates after how many turns to collect the data (0 = never)

  Lattice() {}
  Lattice(const std::string & fname) { read_twiss_table(fname); }
 
  void add(const std::string & ele) {
    lattice.emplace_back(ele);
    edited_lattice = true;
  }

  void read_twiss_table(const std::string & fname) {
    std::ifstream file(fname);
    if (!file.is_open()) {
      throw std::runtime_error("read_twiss_table(): cannot open file "+std::string(fname) );
    }
         
    std::string columns;
    {
      std::map<std::string,std::string> header_map;
      while ( std::getline(file,columns) and (columns.empty() or columns.front() != '*')) {
        //use the initial rows to build the header map, should exit when finds a line starting with *
        std::stringstream ss(columns);
        std::string id;
        std::string name;
        std::string type;
        std::string val;
        ss >> id >> name >> type >> val;
        header_map.emplace(std::move(name), std::move(val));
      }
      energy = stod(header_map.at("ENERGY"));
      mass   = stod(header_map.at("MASS"));  
    }
    if (columns.empty() or columns.front() != '*') {
      throw std::runtime_error("read_twiss_table(): cannot find the field description line in file "+std::string(fname) );
    }

    std::string types;
    while ( std::getline(file,types) and (types.empty() or types.front() != '$')) {
      //discard all the initial rows, should exit when finds a line starting with $
    }
    if (types.empty() or types.front() != '$') {
      throw std::runtime_error("read_twiss_table(): cannot find the type description line in file "+std::string(fname) );
    }

    columns.erase(0, 2);
    types.erase(0, 2);

    std::map<std::string, std::string> map_key_type; //not really needed
    std::map<std::string, std::string> map_key_value;
    std::vector<std::string> keys;

    {
      std::stringstream sc(std::move(columns));
      std::stringstream st(std::move(types));
      while ((sc >> columns) and (st >> types)) {
        keys.push_back(columns);
        map_key_type.emplace(std::move(columns),std::move(types));
      }
    }
    
    //now read the lines and make the elements
    bool first_line = true;
    while ( std::getline(file,columns) ) {
      std::map<std::string, std::string> map_key_value;
      std::stringstream sc(std::move(columns));
      auto it = keys.begin();
      while (sc >> columns) {
        if (columns.front() == '\"' and columns.back() == '\"') {
          columns = columns.substr(1, columns.length()-2);
        }
        map_key_value.emplace(*it++,std::move(columns));
      }
      if (first_line) {
        populate_initial_conditions(map_key_value);
        first_line = false;
      }
      make_element_from_madx_row(map_key_value);
    }
    edited_lattice = true;
  }

  void populate_initial_conditions(const std::map<std::string, std::string> & values) {
    BETX = stod(values.at("BETX")); 
    BETY = stod(values.at("BETY")); 
    ALFX = stod(values.at("ALFX")); 
    ALFY = stod(values.at("ALFY")); 
    DX   = stod(values.at("DX"));
    DY   = stod(values.at("DY"));
    DPX  = stod(values.at("DPX"));
    DPY  = stod(values.at("DPY"));
    X    = stod(values.at("X")); 
    PX   = stod(values.at("PX")); 
    Y    = stod(values.at("Y")); 
    PY   = stod(values.at("PY")); 
  }

  void make_element_from_madx_row(const std::map<std::string, std::string> & values) {
    const auto get_s = [&](const std::string & key) { return values.at(key); };
    const auto get_d = [&](const std::string & key) { return stod(get_s(key)); };
    const auto is_zero = [&](const std::string & key) { return std::abs(get_d(key)) < 1e-15; };

    const std::string & ele_type = get_s("KEYWORD");
    if ( ele_type == "MULTIPOLE" ) {
      const auto KL  = [](const int i){ return "K"+std::to_string(i)+"L" ; };
      const auto KSL = [](const int i){ return "K"+std::to_string(i)+"SL"; };
      int i = 0;
      while ( values.count(KL(i)) ) { //loop on all the strengths in the map
        if ( !is_zero(KL(i)) or !is_zero(KSL(i)) ) {
          if ( i == 0 and !is_zero("K0L") and is_zero("K0SL") ) { 
            // Dipole magnet
            lattice.emplace_back( "Dipole(Angle,L)" );
            replace( lattice.back(), "Angle", get_s("K0L") );
            replace( lattice.back(), "L", get_s("LRAD") );
          } else if ( i == 1 and !is_zero("K1L") and is_zero("K1SL") ) {
            // Quadrupole magnet
            lattice.emplace_back( "Quad(K1L)" );
            replace( lattice.back(), "K1L", get_s("K1L") );
          } else {
            // General Multipole
            lattice.emplace_back( "Multipole(O,KL,KSL)" );
            replace( lattice.back(), "O"  , std::to_string(i) );
            replace( lattice.back(), "KL" , get_s(KL(i))  );
            replace( lattice.back(), "KSL", get_s(KSL(i)) );
          }
        }
        ++i;
      }
    } 
      else if ( ele_type == "HKICKER" ) {
      if ( !is_zero("HKICK") ) {
        lattice.emplace_back( "HKicker(%K)" );
        replace( lattice.back(), "%K", get_s("HKICK") );
      }
    } else if ( ele_type == "VKICKER" ) {
      if ( !is_zero("VKICK") ) {
        lattice.emplace_back( "VKicker(%K)" );
        replace( lattice.back(), "%K", get_s("VKICK") );
      }
    } 
      else if ( ele_type == "MARKER" ) { //drop it silently
    } else if ( ele_type == "RFCAVITY" ) {
       //double VE = get_d ("VOLT") / energy; 
       double VE = 20.0/ energy; 
       //std::cout<<VE<<" "<<get_d("VOLT")<<std::endl;
       lattice.emplace_back( "RF(%f,%VE)");
       //replace( lattice.back(), "%f", get_s("FREQ") );
       replace( lattice.back(), "%f", std::to_string(299.7924) );
       replace( lattice.back(), "%VE", std::to_string(VE) );
    } else {
      if ( !is_zero("L") ) {
        if ( ele_type != "DRIFT" ) {
          std::cerr << "Replacing " << ele_type << " with drift" << std::endl;
        }
        lattice.emplace_back( "Drift(%L)" );
        replace(lattice.back(), "%L", values.at("L"));
      } else {
//        std::cerr << "Dropping zero-length unknown element: "
//                  << ele_type << std::endl;
      }
    }
  }

  void optimise() {
    const auto is_drift = [](const std::string & line) {
      return ( line.find("Drift") != std::string::npos );
    };
    const auto get_drift_length = [](const std::string & line) {
      const size_t start = line.find('(')+1;
      const size_t len   = line.find(')', start) - start;
      return stod(line.substr(start, len));
    };

    size_t old_size, new_size;
    do {
      old_size = lattice.size();
      for (size_t i = 0; i < lattice.size()-1; ++i) {
        std::string & line = lattice[i];
        std::string & next = lattice[i+1];
        //coaleshing contiguous drifts
        if ( is_drift(line) and is_drift(next) ) {
          const double L = get_drift_length(line) + get_drift_length(next);
          lattice[i] = "Drift(%L)";
          replace(lattice[i], "%L", std::to_string(L));
          lattice.erase(lattice.begin()+i+1);
        }
      }
      new_size = lattice.size();
    } while (new_size < old_size);
    edited_lattice = true;
  }

  std::string get_kernel_function(const size_t start, const size_t size) const {
    if (lattice.size() == 0) throw std::runtime_error("The lattice looks empty..");
    std::string res = R"===(
#include "track.h"
extern "C" __global__
void track(const size_t n, Particle* p) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
)===";

    for (size_t i = start; (i < start+size) and (i < lattice.size()); i++) {
//      std::string line(lattice[i]);
//      res += line.insert(line.find('(')+1, "p,");
//      res += ";\n";
      res += lattice[i] + ";\n";
    }
    res += "}}\n";
std::cout << "CUDA KERNEL:\n" << res << std::endl;
    return res;
  }

  size_t get_n_elements() const {
    return lattice.size();
  }

  void compile() {
  Timer("Compiling ptx", [&](){
    const size_t ptx_n_ele = 800;
    if ( edited_lattice ) {
      nvrtcs.clear();
  
      for (size_t i = 0; i < lattice.size(); i+=ptx_n_ele) {
        nvrtcs.emplace_back();
        nvrtcs.back().compile_ptx(get_kernel_function(i, ptx_n_ele));
      }
  
  // CANNOT AVOID SEGFAULT WITH PARALLEL COMPILATION...
  //    std::vector<std::thread> threads; 
  //    for (size_t i = 0; i < lattice.size(); i+=ptx_n_ele) {
  //      nvrtcs.emplace_back();
  //      NVRTC * nvrtc = &(nvrtcs.back());
  //      threads.push_back(std::thread( [&](size_t j, NVRTC * n){
  //        n->compile_ptx(get_kernel_function(j, ptx_n_ele));
  //      }, i, nvrtc ));
  //    }
  //std::cout << "WAITING THREADS" << std::endl;
  //    for (auto & thread: threads) {
  //      thread.join();
  //    }
  //std::cout << "DONE COMPILING" << std::endl;
  
      edited_lattice = false;
    }
  });
  }

  void write_ptx(const std::string & path) {
    size_t i = 0;
    //remove the existing ptx to avoid concatenating old files when reading
    while ( remove((path+std::to_string(i)+".ptx").c_str())==0 ) { ++i; }
    //write the new files
    for (i = 0; i < nvrtcs.size(); ++i) {
      nvrtcs[i].write_ptx(path+std::to_string(i)+".ptx");
    }
  }

  bool file_exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
  }
  void read_ptx(const std::string & path) {
    size_t i = 0;
    while ( file_exists(path+std::to_string(i)+".ptx") ) {
      nvrtcs.emplace_back(path+std::to_string(i)+".ptx");
      i++;
    }
    if ( i == 0 ) {
      throw std::runtime_error("cannot find ptx "+path);
    }
    edited_lattice = false;
  }


  void track(DeviceBunch & b) {
    if ( edited_lattice ) {
      compile();
    }
    Timer("Running", [&](){ 
      for (size_t i = 0; i < n_turns; ++i) {
        if ( collect_tbt_data and i % collect_tbt_data == 0 ) turn_by_turn_data.emplace_back(HostBunch(b));
        for (auto & nvrtc: nvrtcs) {
          nvrtc.run(b.get_data().data());
        }
      }
      if ( collect_tbt_data and n_turns % collect_tbt_data == 0 ) turn_by_turn_data.emplace_back(HostBunch(b));});
  }

  void track(HostBunch & b) {
    DeviceBunch db;
    Timer("Transfering bunch to GPU", [&](){db = b;} );
    track(db);
    Timer("Transfering bunch to CPU", [&](){b = db;} );
  }

  HostBunch make_matched_bunch(const size_t N) const {
    if ( !(BETX > 0.) ) {
      throw std::runtime_error("Need to specify the initial BETX when making a matched bunch");
    }
    if ( !(BETY > 0.) ) {
      throw std::runtime_error("Need to specify the initial BETY when making a matched bunch");
    }
    if ( !(energy > 0.) ) {
      throw std::runtime_error("Need to specify the beam energy when making a matched bunch");
    }

    HostBunch b;

    std::normal_distribution<double> dist_x (0.0,sigma_x());
    std::normal_distribution<double> dist_xp(0.0,sigma_xp());
    std::normal_distribution<double> dist_y (0.0,sigma_y());
    std::normal_distribution<double> dist_yp(0.0,sigma_yp());
 
    // TWISS
    for (size_t i = 0; i < N; ++i) {
      Particle p;

      p.x  = dist_x (bunch_rnd_gen);
      p.y  = dist_y (bunch_rnd_gen);
      p.px = dist_xp(bunch_rnd_gen) - ALFX * p.x/sigma_x() * sigma_xp();
      p.py = dist_yp(bunch_rnd_gen) - ALFY * p.y/sigma_y() * sigma_yp();

      b.particles.emplace_back(p);
    }
    b.set_z(bunch_length);
    b.set_d(bunch_energy_spread);

    // Orbit and Dispersion
    for (size_t i = 0; i < N; ++i) {
      b.x (i) += X  + DX  * b.d(i);
      b.y (i) += Y  + DY  * b.d(i);
      b.px(i) += PX + DPX * b.d(i);
      b.py(i) += PY + DPY * b.d(i);
    }
    return b;
  }

  double geo_emit_x() const {
    return norm_emit_x/beta_gamma(energy,mass);
  }
  
  double geo_emit_y() const {
    return norm_emit_y/beta_gamma(energy,mass);
  }

  double sigma_x() const {
    return sqrt( geo_emit_x() * BETX);
  } 

  double sigma_xp() const {
    return sqrt( geo_emit_x() / BETX);
  } 

  double sigma_y() const {
    return sqrt( geo_emit_y() * BETY);
  }

  double sigma_yp() const {
    return sqrt( geo_emit_y() / BETY);
  } 
  double rel_gamma() const {
    return gamma_rel(energy,mass);
  }
  void clear_tbt_data() {
    turn_by_turn_data.clear();
  }

};

