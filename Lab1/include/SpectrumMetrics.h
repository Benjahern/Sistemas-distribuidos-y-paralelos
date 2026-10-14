#pragma once
#include "ComplexField.h"

struct ParsevalResult {
    double input_energy{};
    double spectrum_energy{}; // Ya dividida por M*N.
    double absolute_error{};
    double relative_error{};
};

// other es la reconstrucción para roundtripError(), o el espectro para parseval().
class SpectrumMetrics {
public:
    SpectrumMetrics(const ComplexField& original, const ComplexField& other);
    double roundtripError() const;
    ParsevalResult parseval(int method = 0) const;
    ParsevalResult parseval(int method, bool use_private) const;
    static double energy(const ComplexField& field, int method, bool use_private);
private:
    const ComplexField& original_;
    const ComplexField& other_;
};
