/*
 * Precision Test
 *
 */

#include "neural/metrics/precision.h"

#include <gtest/gtest.h>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(StatsTest, TestMetricsPrecisionHighCutoff)
{
    float confidenceThreshold = 0.7;
    TTensorPtr l_outputs = Tensor::New({4, 3},
        {
            0.75, 0.15, 0.1, // tp, correct, and confidence is over 0.7
            0.6, 0.2, 0.2,   // fn, incorrect, but our confidence was lower than the threshold
            0.1, 0.5, 0.4,   // fn, incorrect, but our confidence was lower than the threshold
            0.34, 0.33, 0.33 // tn, correct, but our confidence was too low
        });

    // precision = tp / (tp + fp)
    // there are no false positives here so precision == 1.0
    TTensorPtr l_targets = Tensor::New({4, 3},
        {
            1, 0, 0, // tp
            0, 1, 0, // tn
            0, 0, 1, // fn
            1, 0, 0  // tn
        });

    metrics::Precision l_precision;
    l_precision.AddResults(l_outputs, l_targets);

    EXPECT_NEAR(1.0, l_precision.Calculate(confidenceThreshold), 0.001);
}

TEST(StatsTest, TestMetricsPrecisionMedCutoff)
{
    TTensorPtr l_outputs = Tensor::New({5, 3},
        {
            0.75, 0.15, 0.1,
            0.6, 0.2, 0.2,
            0.1, 0.25, 0.65,
            0.1, 0.44, 0.46,
            0.34, 0.33, 0.33
        });

    // precision = tp / (tp + fp)
    // 2 true positives, 1 false positive, precision = 2/3 == 0.666
    TTensorPtr l_targets = Tensor::New({5, 3},
        {
            1, 0, 0, // tp
            0, 1, 0, // fp
            0, 0, 1, // tp
            0, 0, 1, // tn
            1, 0, 0  // tn
        });

    metrics::Precision l_precision;
    l_precision.AddResults(l_outputs, l_targets);

    EXPECT_NEAR(0.666, l_precision.Calculate(0.5), 0.001);
}

TEST(StatsTest, TestMetricsPrecisionLowCutoff)
{
    TTensorPtr l_outputs = Tensor::New({5, 3},
        {
            0.75, 0.15, 0.1,
            0.6, 0.2, 0.2,
            0.1, 0.25, 0.65,
            0.1, 0.44, 0.46,
            0.34, 0.33, 0.33
        });

    // precision = tp / (tp + fp)
    // 4 true positives, 1 false positive, precision = 4/5 == 0.8
    TTensorPtr l_targets = Tensor::New({5, 3},
        {
            1, 0, 0, // tp
            0, 1, 0, // fp
            0, 0, 1, // tp
            0, 0, 1, // tp
            1, 0, 0  // tp
        });

    metrics::Precision l_precision;
    l_precision.AddResults(l_outputs, l_targets);

    EXPECT_NEAR(0.8, l_precision.Calculate(0.1), 0.001);
}
