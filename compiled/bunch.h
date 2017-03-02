#ifndef BUNCH_H
#define BUNCH_H

#include <vector>
#include <random>

#include "nvrtc_wrap.h"
#include "particle.h"

struct HostBunch;

std::default_random_engine bunch_rnd_gen(56789);

class DeviceBunch {
  friend class HostBunch;

  CUdeviceptr particles;
  size_t bufferSize = 0;
  size_t np; //used only to pass out the void* in the args method

 public: 
  DeviceBunch() {}
  DeviceBunch(const size_t N) { alloc(N); }
  DeviceBunch(const HostBunch & hb);
  ~DeviceBunch() { dealloc(); }

  size_t n() const {
    return bufferSize/sizeof(Particle);
  }

  void alloc(const size_t N) {
    if ( bufferSize != 0 ) dealloc();
    bufferSize = N*sizeof(Particle);
    CUDA_SAFE_CALL(cuMemAlloc(&particles,  bufferSize));
  }

  void dealloc() {
    CUDA_SAFE_CALL(cuMemFree(particles));
    bufferSize = 0;
  }

  std::vector<void*> get_data() {
    np = n();
    std::vector<void*> args;
    args.push_back( (void*) &np );
    args.push_back( (void*) &particles);
    return args;
  }

  void operator=(const HostBunch & hb);
};

struct HostBunch {
  std::vector<Particle> particles;

  HostBunch() {}
  HostBunch(const size_t N):
    particles(N,Particle())
  {std::cout << "HostBunch::HostBunch " << N << " "<< size()<< std::endl; }

  void set_z(const double sigma_z, const double mean_z = 0.) {
    std::normal_distribution<double> dist(mean_z,sigma_z);
    for (size_t i = 0; i < size(); ++i) {
      particles[i].sigma = dist(bunch_rnd_gen);
    }
  }

  void set_d(const double sigma_d, const double mean_d = 0.) {
    std::normal_distribution<double> dist(mean_d,sigma_d);
    for (size_t i = 0; i < size(); ++i) {
      particles[i].delta = dist(bunch_rnd_gen);
    }
  }

  HostBunch(const HostBunch & o) = default;
  HostBunch(HostBunch && o) = default;
  HostBunch & operator=(const HostBunch & o) = default;
  HostBunch & operator=(HostBunch && o) = default;
 
  bool operator==(const HostBunch & o) {
//    return ( particles == o.particles );
    return (this == &o);
  }

  HostBunch(const DeviceBunch & db) { copyFromDeviceBunch(db); }

  void copyFromDeviceBunch(const DeviceBunch & db) {
    particles.resize(db.n());
    CUDA_SAFE_CALL(cuMemcpyDtoH(particles.data(), db.particles, db.bufferSize));
  }

  void copyToDeviceBunch(DeviceBunch & db) const {
    db.alloc(size());
    CUDA_SAFE_CALL(cuMemcpyHtoD(db.particles, particles.data(), db.bufferSize));
  }
  size_t size() const {
    return particles.size();
  }

  void operator=(const DeviceBunch & db) {
    copyFromDeviceBunch(db);
  }

  double & x (const size_t i) { return particles[i].x ; }
  double & px(const size_t i) { return particles[i].px; }
  double & y (const size_t i) { return particles[i].y ; }
  double & py(const size_t i) { return particles[i].py; }
  double & z (const size_t i) { return particles[i].sigma; }
  double & d (const size_t i) { return particles[i].delta; }
};

void DeviceBunch::operator=(const HostBunch & hb) {
  hb.copyToDeviceBunch(*this);
}

DeviceBunch::DeviceBunch(const HostBunch & hb) {
  hb.copyToDeviceBunch(*this);
}

std::vector<double> get_x_from_bunches (std::vector<HostBunch> &vb, const size_t particle_id) {
  std::vector <double> res;
  for (auto & b: vb) {
    res.push_back(b.x(particle_id));
  }
  return res;
}

std::vector<double> get_px_from_bunches (std::vector<HostBunch> &vb, const size_t particle_id) {
  std::vector <double> res;
  for (auto & b: vb) {
    res.push_back(b.px(particle_id));
  }
  return res;
}

std::vector<double> get_y_from_bunches (std::vector<HostBunch> &vb, const size_t particle_id) {
  std::vector <double> res;
  for (auto & b: vb) {
    res.push_back(b.y(particle_id));
  }
  return res;
}

std::vector<double> get_py_from_bunches (std::vector<HostBunch> &vb, const size_t particle_id) {
  std::vector <double> res;
  for (auto & b: vb) {
    res.push_back(b.py(particle_id));
  }
  return res;
}

std::vector<double> get_z_from_bunches (std::vector<HostBunch> &vb, const size_t particle_id) {
  std::vector <double> res;
  for (auto & b: vb) {
    res.push_back(b.z(particle_id));
  }
  return res;
}

std::vector<double> get_d_from_bunches (std::vector<HostBunch> &vb, const size_t particle_id) {
  std::vector <double> res;
  for (auto & b: vb) {
    res.push_back(b.d(particle_id));
  }
  return res;
}

#endif
