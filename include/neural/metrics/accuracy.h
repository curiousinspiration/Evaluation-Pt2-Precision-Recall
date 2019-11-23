/*
 * Calculate accuracy given predictions and targets stored in parent class
 * 
 */

#pragma once

#include "neural/metrics/metric.h"

namespace neural
{

namespace metrics
{

class Accuracy : public Metric
{

public:
    Accuracy(size_t a_runningAvgLen = 1000);
    virtual ~Accuracy() {};

    // name for logging / debugging
    virtual const std::string& GetName() const override;

    // calculate the metric
    virtual float Calculate(float a_confidence = 0.0) const override;

private:
    static const std::string NAME;

};

} // namespace metric

} // namespace neural
