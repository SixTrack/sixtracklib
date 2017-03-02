#pragma once

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

template <typename T = const char *>
std::vector<T> vecStr2vec(const std::vector<std::string> & v) {
  std::vector<T> out;
  for (const auto & s: v) {
    out.push_back(s.c_str());
  }
  return out;
}

template <typename VEC, typename SCAL>
void vector_scalar_add(VEC & v, const SCAL s) {
  for (auto & i: v) i += s;
}

template <typename VEC, typename SCAL>
void vector_scalar_mul(VEC & v, const SCAL s) {
  for (auto & i: v) i *= s;
}

template <typename VEC>
double sum(const VEC & v) {
  return sum = std::accumulate(std::begin(v), std::end(v), 0.0);
}

template <typename VEC>
double avg(const VEC & v) {
  return sum(v) / v.size();
}

template <typename VEC>
void sig(const VEC & v) {
  const double m = avg(v);

  double accum = 0.0;
  std::for_each (std::begin(v), std::end(v), [&](const double d) {
    accum += (d - m) * (d - m);
  });

  return sqrt(accum / (v.size()-1));
}

template <typename VEC>
double cov(const VEC & v1, const VEC & v2) {
  if ( v1.size() != v2.size() ) {
    throw std::runtime_error
      ("Cannot compute the covariance of two containers of different sizes");
  }
  const double m1 = avg(v1);
  const double m2 = avg(v2);

  double accum = 0.0;
  for (size_t i = 0; i < v1.size(); ++i) {
    accum += (v1[i] - m1) * (v2[i] - m2);
  };

  return sqrt(accum / (v1.size()-1));
}

double gamma_rel(const double energy, const double mass) {
  return energy/mass; //total energy
}

double beta_gamma(const double energy, const double mass) {
  const auto gamma = gamma_rel(energy, mass);
  return (1.+gamma*gamma)/gamma;
}

//replace "from" with "to" in string "str"
bool replace(std::string& str, const std::string& from, const std::string& to) {
  size_t start_pos = str.find(from);
  if(start_pos == std::string::npos)
      return false;
  str.replace(start_pos, from.length(), to);
  return true;
}

