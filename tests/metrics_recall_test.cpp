/*
 * Recall Test
 *
 */

#include "neural/metrics/recall.h"

#include <gtest/gtest.h>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(StatsTest, TestMetricsRecallHighCutoff)
{
    TTensorPtr l_outputs = Tensor::New({5, 3},
        {
            0.75, 0.15, 0.1,
            0.6, 0.2, 0.2,
            0.1, 0.25, 0.65,
            0.1, 0.44, 0.46,
            0.34, 0.33, 0.33
        });

    // Recall = tp / (tp + fn)
    // there are 3 false negatives and 1 true positive Recall == 1/4 == 0.25
    TTensorPtr l_targets = Tensor::New({5, 3},
        {
            1, 0, 0, // tp
            0, 1, 0, // fn
            0, 0, 1, // fn
            0, 0, 1, // tn
            1, 0, 0  // fn
        });

    metrics::Recall l_recall;
    l_recall.AddResults(l_outputs, l_targets);

    EXPECT_NEAR(0.25, l_recall.Calculate(0.7), 0.001);
}

TEST(StatsTest, TestMetricsRecallLowCutoff)
{
    TTensorPtr l_outputs = Tensor::New({5, 3},
        {
            0.75, 0.15, 0.1,
            0.6, 0.2, 0.2,
            0.1, 0.25, 0.65,
            0.1, 0.44, 0.46,
            0.34, 0.33, 0.33
        });

    // Recall = tp / (tp + fn)
    // 4 true positives, 0 false negatives, Recall = 4/4 == 1.0
    TTensorPtr l_targets = Tensor::New({5, 3},
        {
            1, 0, 0, // tp
            0, 1, 0, // fp
            0, 0, 1, // tp
            0, 0, 1, // tp
            1, 0, 0  // tp
        });

    metrics::Recall l_recall;
    l_recall.AddResults(l_outputs, l_targets);

    EXPECT_NEAR(1.0, l_recall.Calculate(0.1), 0.001);
}
