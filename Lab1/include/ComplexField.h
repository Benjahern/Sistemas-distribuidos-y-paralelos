#ifndef COMPLEX_FIELD_H
#define COMPLEX_FIELD_H

#include <cstddef>
#include <complex>
#include <vector>

/**
 * @brief Rol 1: Modelo de datos para grilla compleja bidimensional M x N.
 *
 */
class ComplexField {
public:
    ComplexField() = default;
    ComplexField(size_t rows, size_t cols);
    ComplexField(size_t rows, size_t cols, const std::complex<double>& initial_value);
    ~ComplexField() = default;

    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }
    size_t size() const noexcept { return data_.size(); }

    std::complex<double>& at(size_t r, size_t c) { return data_[r * cols_ + c]; }
    const std::complex<double>& at(size_t r, size_t c) const { return data_[r * cols_ + c]; }

    std::complex<double>& operator()(size_t r, size_t c) { return data_[r * cols_ + c]; }
    const std::complex<double>& operator()(size_t r, size_t c) const { return data_[r * cols_ + c]; }

    std::complex<double>* data() noexcept { return data_.data(); }
    const std::complex<double>* data() const noexcept { return data_.data(); }

    std::complex<double>* row(size_t r) noexcept { return data_.data() + (r * cols_); }
    const std::complex<double>* row(size_t r) const noexcept { return data_.data() + (r * cols_); }

private:
    size_t rows_{0};
    size_t cols_{0};
    std::vector<std::complex<double>> data_;
};

#endif // COMPLEX_FIELD_H
