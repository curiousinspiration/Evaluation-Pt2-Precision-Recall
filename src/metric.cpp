/*
 * Metric Implementation
 *
 */

#include "neural/metrics/metric.h"

using namespace std;

namespace neural
{

namespace metrics
{

Metric::Metric(size_t a_runningAvgLen)
    : m_runningAvgLen(a_runningAvgLen)
{

}

void Metric::AddResults(
    const TTensorPtr& a_outputs, const TTensorPtr& a_targets)
{
    m_outputs.push_back(a_outputs);
    m_targets.push_back(a_targets);

    while (m_outputs.size() > m_runningAvgLen) {
        m_outputs.pop_front();
    }

    while (m_targets.size() > m_runningAvgLen) {
        m_targets.pop_front();
    }
}

size_t Metric::p_IterAndCountWithFn(
    std::function<bool(size_t, size_t, float, float)> a_comparatorFn,
    float a_confidence) const
{
    // total number of occurences
    size_t l_retVal = 0;
    // iterate over all results in the deque
    for (size_t i = 0; i < m_targets.size(); ++i)
    {
        // get the result outputs, and targets
        const TTensorPtr& l_outputs = m_outputs.at(i);
        const TTensorPtr& l_targets = m_targets.at(i);
        
        // remember, this is probably the output of a batch of predictions
        // iterate over rows in predictions / targets
        for (size_t j = 0; j < l_outputs->Shape().at(0); ++j)
        {
            size_t l_targetIdx = l_targets->GetRow(j)->MaxIdx();
            size_t l_predIdx = l_outputs->GetRow(j)->MaxIdx();
            float l_predVal = l_outputs->GetRow(j)->MaxVal();

            if (a_comparatorFn(l_targetIdx, l_predIdx, l_predVal, a_confidence))
            {
                ++l_retVal;
            }
        }
    }
    return l_retVal;
}

size_t Metric::p_CalcNumTruePositives(
    float a_confidenceLevel) const
{
    using namespace std::placeholders;
    auto fp = std::bind(&Metric::p_ExampleIsTruePositive, this, _1, _2, _3, _4);
    return p_IterAndCountWithFn(fp, a_confidenceLevel);
}

size_t Metric::p_CalcNumTrueNegatives(
    float a_confidenceLevel) const
{
    using namespace std::placeholders;
    auto fp = std::bind(&Metric::p_ExampleIsTrueNegative, this, _1, _2, _3, _4);
    return p_IterAndCountWithFn(fp, a_confidenceLevel);
}

size_t Metric::p_CalcNumFalsePositives(
    float a_confidenceLevel) const
{
    using namespace std::placeholders;
    auto fp = std::bind(&Metric::p_ExampleIsFalsePositive, this, _1, _2, _3, _4);
    return p_IterAndCountWithFn(fp, a_confidenceLevel);
}

size_t Metric::p_CalcNumFalseNegatives(
    float a_confidenceLevel) const
{
    using namespace std::placeholders;
    auto fp = std::bind(&Metric::p_ExampleIsFalseNegative, this, _1, _2, _3, _4);
    return p_IterAndCountWithFn(fp, a_confidenceLevel);
}

bool Metric::p_ExampleIsTruePositive(
    size_t a_targetIdx, size_t a_predIdx,
    float a_guessConfidence, float a_confidenceCutoff) const
{
    // correct, and above confidence threshold
    return (a_targetIdx == a_predIdx) and (a_guessConfidence > a_confidenceCutoff);
}

bool Metric::p_ExampleIsFalsePositive(
    size_t a_targetIdx, size_t a_predIdx,
    float a_guessConfidence, float a_confidenceCutoff) const
{
    // incorrect, and above confidence threshold
    return (a_targetIdx != a_predIdx) and (a_guessConfidence > a_confidenceCutoff);
}

bool Metric::p_ExampleIsTrueNegative(
    size_t a_targetIdx, size_t a_predIdx,
    float a_guessConfidence, float a_confidenceCutoff) const
{
    // incorrect, but below confidence threshold
    return (a_targetIdx != a_predIdx) and (a_guessConfidence < a_confidenceCutoff);
}

bool Metric::p_ExampleIsFalseNegative(
    size_t a_targetIdx, size_t a_predIdx,
    float a_guessConfidence, float a_confidenceCutoff) const
{
    // correct, but below confidence threshol
    return (a_targetIdx == a_predIdx) and (a_guessConfidence < a_confidenceCutoff);
}

} // namespace metrics

} // namespace neural
