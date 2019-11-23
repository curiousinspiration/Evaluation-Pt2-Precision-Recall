/*
 * metrics::Metric is the base class for Precision/Recall that
 * knows how to calculate true positives, false positives, true negatives,
 * and false negatives
 * 
 */

#pragma once

#include "neural/math/tensor.h"

#include <deque>
#include <functional>

namespace neural
{

namespace metrics
{

class Metric
{

public:
    Metric(size_t a_runningAvgLen = 1000);

    // name for logging / debugging
    virtual const std::string& GetName() const = 0;

    // calculate the metric
    virtual float Calculate(float a_confidenceLevel = 0.5) const = 0;

    // Store results internally so we can calculate over time
    void AddResults(
        const TTensorPtr& a_outputs, const TTensorPtr& a_targets);

protected:
    // Keep track of `m_runningAvgLen` examples so we can calculate given
    // last n examples
    size_t m_runningAvgLen;
    // We will keep a state of last `m_runningAvgLen` outputs and targets
    std::deque<const TTensorPtr> m_outputs;
    std::deque<const TTensorPtr> m_targets;

    size_t p_IterAndCountWithFn(
        std::function<bool(size_t, size_t, float, float)> a_comparatorFn,
        float a_confidence) const;

    size_t p_CalcNumTruePositives(
        float a_confidence) const;
    size_t p_CalcNumTrueNegatives(
        float a_confidence) const;
    size_t p_CalcNumFalsePositives(
        float a_confidence) const;
    size_t p_CalcNumFalseNegatives(
        float a_confidence) const;

    bool p_ExampleIsTruePositive(
        size_t a_targetIdx, size_t a_predIdx,
        float a_guessConfidence, float a_confidenceCutoff) const;
    bool p_ExampleIsFalsePositive(
        size_t a_targetIdx, size_t a_predIdx,
        float a_guessConfidence, float a_confidenceCutoff) const;
    bool p_ExampleIsTrueNegative(
        size_t a_targetIdx, size_t a_predIdx,
        float a_guessConfidence, float a_confidenceCutoff) const;
    bool p_ExampleIsFalseNegative(
        size_t a_targetIdx, size_t a_predIdx,
        float a_guessConfidence, float a_confidenceCutoff) const;

};

} // namespace metrics

} // namespace neural
