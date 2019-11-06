/*
 * Accuracy Test
 *
 */

#include "neural/metrics/accuracy.h"

#include <gtest/gtest.h>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(StatsTest, TestMetricsAccuracy)
{
    TTensorPtr l_outputs = Tensor::New({4, 3},
        {
            0.75, 0.15, 0.1,
            0.6, 0.2, 0.2,
            0.1, 0.5, 0.4,
            0.34, 0.33, 0.33
        });

    TTensorPtr l_targets = Tensor::New({4, 3},
        {
            1, 0, 0, // true
            0, 1, 0, // false
            0, 0, 1, // false
            1, 0, 0  // true
        });

    // 2/4 = 0.5
    metrics::Accuracy l_accuracy;
    l_accuracy.AddResults(l_outputs, l_targets);

    EXPECT_NEAR(0.5, l_accuracy.Calculate(), 0.001);
}
