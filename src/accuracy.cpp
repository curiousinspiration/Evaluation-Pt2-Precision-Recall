/*
 * Accuracy Metric Implementation
 *
 */

#include "neural/metrics/accuracy.h"

#include <glog/logging.h>

using namespace std;

namespace neural
{

namespace metrics
{

const std::string Accuracy::NAME = "accuracy";

Accuracy::Accuracy(size_t a_runningAvgLen)
    : Metric(a_runningAvgLen)
{

}

// name for logging / debugging
const std::string& Accuracy::GetName() const
{
    return NAME;
}

// calculate the metric
float Accuracy::Calculate(float a_confidenceLevel) const
{
    float l_tp = static_cast<float>(p_CalcNumTruePositives(0.0));
    float l_fp = static_cast<float>(p_CalcNumFalsePositives(0.0));
    float l_tn = static_cast<float>(p_CalcNumTrueNegatives(0.0));
    float l_fn = static_cast<float>(p_CalcNumFalseNegatives(0.0));

    // avoid NaN
    if (0.0f == l_tp and 0.0 == l_tn and 0.0f == l_fp and 0.0f == l_fn)
    {
        return 0.0f;
    }

    return (l_tp + l_tn) / (l_tp + l_tn + l_fp + l_fn);
}

} // namespace metric

} // namespace neural
