#include "Butterfly1D.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

Butterfly1D::Butterfly1D(std::complex<double> *data, size_t n, bool inverse)
    : data_(data), n_(n), inverse_(inverse) {}

void Butterfly1D::setData(std::complex<double> *data, size_t n, bool inverse) {
  data_ = data;
  n_ = n;
  inverse_ = inverse;
}

size_t Butterfly1D::bitReverse(size_t i, size_t log2n) {
  size_t rev = 0;
  for (size_t bit = 0; bit < log2n; ++bit) {
    if ((i >> bit) & 1) {
      rev |= (size_t(1) << (log2n - 1 - bit));
    }
  }
  return rev;
}

void Butterfly1D::transform(int layout, int schedule_type, int chunk_size) {
  if (layout == 1) {
    transformStockham(schedule_type, chunk_size, false);
  } else {
    transformCooleyTukey(schedule_type, chunk_size, false);
  }
}

void Butterfly1D::transform() { transform(0, 0, 0); }

void Butterfly1D::transform(int schedule_type) {
  transform(0, schedule_type, 0);
}

void Butterfly1D::transform(int schedule_type, int chunk_size) {
  transform(0, schedule_type, chunk_size);
}

void Butterfly1D::transformCollapse() { transformStockham(0, 0, true); }

void Butterfly1D::transformStages() { transformCooleyTukey(0, 0, true); }

void Butterfly1D::transformCooleyTukey(int schedule_type, int chunk_size,
                                       bool use_single_region) {
  if (n_ <= 1 || !data_)
    return;

  size_t p = 0;
  while ((size_t(1) << p) < n_)
    ++p;

  // Bit-reversal permutation (in-place)
  for (size_t i = 0; i < n_; ++i) {
    size_t rev = bitReverse(i, p);
    if (i < rev) {
      std::swap(data_[i], data_[rev]);
    }
  }

  const double pi = 3.14159265358979323846;
  const double sign = inverse_ ? 1.0 : -1.0;
  const size_t total_bf = n_ / 2;

  if (use_single_region) {
// Demostración de transformStages(): única región paralela con barrera entre
// etapas
#pragma omp parallel default(none) shared(p, n_, sign, pi, total_bf)
    {
      for (size_t s = 1; s <= p; ++s) {
        size_t m = size_t(1) << s;
        size_t m2 = m >> 1;

#pragma omp for schedule(static)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          size_t group_idx = idx / m2;
          size_t j = idx % m2;
          size_t k = group_idx * m;
          std::complex<double> w =
              std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) /
                                  static_cast<double>(m));
          std::complex<double> u = data_[k + j];
          std::complex<double> v = w * data_[k + j + m2];
          data_[k + j] = u + v;
          data_[k + j + m2] = u - v;
        }
#pragma omp barrier
      }
    }
    return;
  }

  for (size_t s = 1; s <= p; ++s) {
    size_t m = size_t(1) << s;
    size_t m2 = m >> 1;

    if (schedule_type == 1) { // dynamic
      if (chunk_size > 0) {
#pragma omp parallel for schedule(dynamic, chunk_size) default(none)           \
    shared(m, m2, sign, pi, total_bf, chunk_size)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          size_t group_idx = idx / m2;
          size_t j = idx % m2;
          size_t k = group_idx * m;
          std::complex<double> w =
              std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) /
                                  static_cast<double>(m));
          std::complex<double> u = data_[k + j];
          std::complex<double> v = w * data_[k + j + m2];
          data_[k + j] = u + v;
          data_[k + j + m2] = u - v;
        }
      } else {
#pragma omp parallel for schedule(dynamic) default(none)                       \
    shared(m, m2, sign, pi, total_bf)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          size_t group_idx = idx / m2;
          size_t j = idx % m2;
          size_t k = group_idx * m;
          std::complex<double> w =
              std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) /
                                  static_cast<double>(m));
          std::complex<double> u = data_[k + j];
          std::complex<double> v = w * data_[k + j + m2];
          data_[k + j] = u + v;
          data_[k + j + m2] = u - v;
        }
      }
    } else if (schedule_type == 2) { // guided
      if (chunk_size > 0) {
#pragma omp parallel for schedule(guided, chunk_size) default(none)            \
    shared(m, m2, sign, pi, total_bf, chunk_size)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          size_t group_idx = idx / m2;
          size_t j = idx % m2;
          size_t k = group_idx * m;
          std::complex<double> w =
              std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) /
                                  static_cast<double>(m));
          std::complex<double> u = data_[k + j];
          std::complex<double> v = w * data_[k + j + m2];
          data_[k + j] = u + v;
          data_[k + j + m2] = u - v;
        }
      } else {
#pragma omp parallel for schedule(guided) default(none)                        \
    shared(m, m2, sign, pi, total_bf)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          size_t group_idx = idx / m2;
          size_t j = idx % m2;
          size_t k = group_idx * m;
          std::complex<double> w =
              std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) /
                                  static_cast<double>(m));
          std::complex<double> u = data_[k + j];
          std::complex<double> v = w * data_[k + j + m2];
          data_[k + j] = u + v;
          data_[k + j + m2] = u - v;
        }
      }
    } else { // static (default)
      if (chunk_size > 0) {
#pragma omp parallel for schedule(static, chunk_size) default(none)            \
    shared(m, m2, sign, pi, total_bf, chunk_size)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          size_t group_idx = idx / m2;
          size_t j = idx % m2;
          size_t k = group_idx * m;
          std::complex<double> w =
              std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) /
                                  static_cast<double>(m));
          std::complex<double> u = data_[k + j];
          std::complex<double> v = w * data_[k + j + m2];
          data_[k + j] = u + v;
          data_[k + j + m2] = u - v;
        }
      } else {
#pragma omp parallel for schedule(static) default(none)                        \
    shared(m, m2, sign, pi, total_bf)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          size_t group_idx = idx / m2;
          size_t j = idx % m2;
          size_t k = group_idx * m;
          std::complex<double> w =
              std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) /
                                  static_cast<double>(m));
          std::complex<double> u = data_[k + j];
          std::complex<double> v = w * data_[k + j + m2];
          data_[k + j] = u + v;
          data_[k + j + m2] = u - v;
        }
      }
    }
  }
}

void Butterfly1D::transformStockham(int schedule_type, int chunk_size,
                                    bool use_collapse) {
  if (n_ <= 1 || !data_)
    return;

  size_t p = 0;
  while ((size_t(1) << p) < n_)
    ++p;

  std::vector<std::complex<double>> src(data_, data_ + n_);
  std::vector<std::complex<double>> dst(n_);

  const double pi = 3.14159265358979323846;
  const double sign = inverse_ ? 1.0 : -1.0;

  for (size_t s = 0; s < p; ++s) {
    size_t l = size_t(1) << s;
    size_t g = n_ / (size_t(1) << (s + 1));
    size_t m = size_t(1) << (s + 1);

    if (use_collapse) {
#pragma omp parallel for collapse(2) default(none)                             \
    shared(src, dst, l, g, m, sign, pi)
      for (size_t k = 0; k < l; ++k) {
        for (size_t j = 0; j < g; ++j) {
          std::complex<double> omega =
              std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) /
                                  static_cast<double>(m));
          size_t i0 = j + 2 * k * g;
          size_t i1 = i0 + g;
          size_t o0 = j + k * g;
          size_t o1 = o0 + n_ / 2;
          std::complex<double> u = src[i0];
          std::complex<double> v = omega * src[i1];
          dst[o0] = u + v;
          dst[o1] = u - v;
        }
      }
    } else {
      if (schedule_type == 1) { // dynamic
        if (chunk_size > 0) {
#pragma omp parallel for schedule(dynamic, chunk_size) default(none)           \
    shared(src, dst, l, g, m, sign, pi, chunk_size)
          for (size_t k = 0; k < l; ++k) {
            for (size_t j = 0; j < g; ++j) {
              std::complex<double> omega =
                  std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) /
                                      static_cast<double>(m));
              size_t i0 = j + 2 * k * g;
              size_t i1 = i0 + g;
              size_t o0 = j + k * g;
              size_t o1 = o0 + n_ / 2;
              std::complex<double> u = src[i0];
              std::complex<double> v = omega * src[i1];
              dst[o0] = u + v;
              dst[o1] = u - v;
            }
          }
        } else {
#pragma omp parallel for schedule(dynamic) default(none)                       \
    shared(src, dst, l, g, m, sign, pi)
          for (size_t k = 0; k < l; ++k) {
            for (size_t j = 0; j < g; ++j) {
              std::complex<double> omega =
                  std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) /
                                      static_cast<double>(m));
              size_t i0 = j + 2 * k * g;
              size_t i1 = i0 + g;
              size_t o0 = j + k * g;
              size_t o1 = o0 + n_ / 2;
              std::complex<double> u = src[i0];
              std::complex<double> v = omega * src[i1];
              dst[o0] = u + v;
              dst[o1] = u - v;
            }
          }
        }
      } else if (schedule_type == 2) { // guided
        if (chunk_size > 0) {
#pragma omp parallel for schedule(guided, chunk_size) default(none)            \
    shared(src, dst, l, g, m, sign, pi, chunk_size)
          for (size_t k = 0; k < l; ++k) {
            for (size_t j = 0; j < g; ++j) {
              std::complex<double> omega =
                  std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) /
                                      static_cast<double>(m));
              size_t i0 = j + 2 * k * g;
              size_t i1 = i0 + g;
              size_t o0 = j + k * g;
              size_t o1 = o0 + n_ / 2;
              std::complex<double> u = src[i0];
              std::complex<double> v = omega * src[i1];
              dst[o0] = u + v;
              dst[o1] = u - v;
            }
          }
        } else {
#pragma omp parallel for schedule(guided) default(none)                        \
    shared(src, dst, l, g, m, sign, pi)
          for (size_t k = 0; k < l; ++k) {
            for (size_t j = 0; j < g; ++j) {
              std::complex<double> omega =
                  std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) /
                                      static_cast<double>(m));
              size_t i0 = j + 2 * k * g;
              size_t i1 = i0 + g;
              size_t o0 = j + k * g;
              size_t o1 = o0 + n_ / 2;
              std::complex<double> u = src[i0];
              std::complex<double> v = omega * src[i1];
              dst[o0] = u + v;
              dst[o1] = u - v;
            }
          }
        }
      } else { // static (default)
        if (chunk_size > 0) {
#pragma omp parallel for schedule(static, chunk_size) default(none)            \
    shared(src, dst, l, g, m, sign, pi, chunk_size)
          for (size_t k = 0; k < l; ++k) {
            for (size_t j = 0; j < g; ++j) {
              std::complex<double> omega =
                  std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) /
                                      static_cast<double>(m));
              size_t i0 = j + 2 * k * g;
              size_t i1 = i0 + g;
              size_t o0 = j + k * g;
              size_t o1 = o0 + n_ / 2;
              std::complex<double> u = src[i0];
              std::complex<double> v = omega * src[i1];
              dst[o0] = u + v;
              dst[o1] = u - v;
            }
          }
        } else {
#pragma omp parallel for schedule(static) default(none)                        \
    shared(src, dst, l, g, m, sign, pi)
          for (size_t k = 0; k < l; ++k) {
            for (size_t j = 0; j < g; ++j) {
              std::complex<double> omega =
                  std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) /
                                      static_cast<double>(m));
              size_t i0 = j + 2 * k * g;
              size_t i1 = i0 + g;
              size_t o0 = j + k * g;
              size_t o1 = o0 + n_ / 2;
              std::complex<double> u = src[i0];
              std::complex<double> v = omega * src[i1];
              dst[o0] = u + v;
              dst[o1] = u - v;
            }
          }
        }
      }
    }
    src.swap(dst);
  }

  for (size_t i = 0; i < n_; ++i) {
    data_[i] = src[i];
  }
}

std::vector<std::complex<double>> Butterfly1D::initTwiddlesSingle() const {
  std::vector<std::complex<double>> twiddles(n_);
  const double pi = 3.14159265358979323846;
  const double sign = inverse_ ? 1.0 : -1.0;

#pragma omp parallel default(none) shared(twiddles, sign, pi)
  {
#pragma omp single
    {
      for (size_t k = 0; k < n_; ++k) {
        twiddles[k] = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) /
                                          static_cast<double>(n_));
      }
    }
  }
  return twiddles;
}

void Butterfly1D::accumulateFirstprivate(double base_val) {
  if (!data_ || n_ == 0)
    return;
  double acc = base_val;
#pragma omp parallel for firstprivate(acc) default(none) shared(data_)
  for (size_t i = 0; i < n_; ++i) {
    data_[i] += acc;
  }
}

int Butterfly1D::stageIndexLastprivate() const {
  size_t p = 0;
  while ((size_t(1) << p) < n_)
    ++p;
  int last_stage = 0;
#pragma omp parallel for lastprivate(last_stage) default(none) shared(p)
  for (int s = 1; s <= static_cast<int>(p); ++s) {
    last_stage = s;
  }
  return last_stage;
}
