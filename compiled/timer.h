#pragma once

#include <chrono>

template <typename Func>
void Timer(const std::string & message, const Func & f) {
  using namespace std::chrono;
  std::cerr << message << "... ";
  auto start = high_resolution_clock::now();
  f();
  std::cerr << "done in "
          << duration_cast<milliseconds>(high_resolution_clock::now() - start).count()*1e-3
          << " s" << std::endl;
}

