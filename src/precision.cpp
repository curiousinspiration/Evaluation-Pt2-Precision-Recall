/*
 * Precision Metric Implementation
 *
 */

#include "neural/metrics/precision.h"

using namespace std;

namespace neural
{

namespace metrics
{

const std::string Precision::NAME = "precision";

Precision::Precision(size_t a_runningAvgLen)
    : Metric(a_runningAvgLen)
{

}

// name for logging / debugging
const std::string& Precision::GetName() const
{
    return NAME;
}

// calculate the metric
float Precision::Calculate(float a_confidenceLevel) const
{
    float l_tp = static_cast<float>(p_CalcNumTruePositives(a_confidenceLevel));
    float l_fp = static_cast<float>(p_CalcNumFalsePositives(a_confidenceLevel));

    if (0.0f == l_tp and 0.0f == l_fp)
    {
        return 0.0f;
    }

    return l_tp / (l_tp + l_fp);
}

} // namespace metric

} // namespace neural
