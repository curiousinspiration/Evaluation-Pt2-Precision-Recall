/*
 * Recall Metric Implementation
 *
 */

#include "neural/metrics/recall.h"

using namespace std;

namespace neural
{

namespace metrics
{

const std::string Recall::NAME = "recall";

Recall::Recall(size_t a_runningAvgLen)
    : Metric(a_runningAvgLen)
{

}

// name for logging / debugging
const std::string& Recall::GetName() const
{
    return NAME;
}

// calculate the metric
float Recall::Calculate(float a_confidenceLevel) const
{
    float l_tp = static_cast<float>(p_CalcNumTruePositives(a_confidenceLevel));
    float l_fn = static_cast<float>(p_CalcNumFalseNegatives(a_confidenceLevel));

    if (0.0f == l_tp and 0.0f == l_fn)
    {
        return 0.0f;
    }
    return l_tp / (l_tp + l_fn);
}

} // namespace metric

} // namespace neural
