/*
 * Calculate recall given predictions and targets stored in parent class
 * 
 */

#pragma once

#include "neural/metrics/metric.h"

namespace neural
{

namespace metrics
{

class Recall : public Metric
{
public:
    Recall(size_t a_runningAvgLen = 1000);

    // name for logging / debugging
    virtual const std::string& GetName() const;

    // calculate the metric
    virtual float Calculate(float a_confidenceLevel = 0.5) const;
private:

    static const std::string NAME;
};

} // namespace metric

} // namespace neural
