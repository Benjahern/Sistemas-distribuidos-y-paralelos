#pragma once
#include "ComplexField.h"
#include <string>
#include <vector>

class Visualizer {
public:
    static std::vector<double> magnitudeNoWait(const ComplexField& spectrum);
    static void writeSpectrum(const ComplexField& spectrum, const std::string& path);
};
