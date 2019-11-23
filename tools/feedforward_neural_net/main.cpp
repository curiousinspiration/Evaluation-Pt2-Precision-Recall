/*
 * Example tool training feed forward neural network on mnist data
 *
 */


#include "neural/data/mnist_dataloader.h"
#include "neural/layers/linear_layer.h"
#include "neural/layers/relu_layer.h"
#include "neural/layers/softmax_layer.h"
#include "neural/loss/cross_entropy_loss.h"
#include "neural/loss/mean_squared_error_loss.h"
#include "neural/metrics/precision.h"
#include "neural/metrics/recall.h"
#include "neural/metrics/accuracy.h"

#include <glog/logging.h>
#include <map>
#include <math.h>

using namespace neural;
using namespace std;

float CalcAverage(const vector<float>& vals)
{
    float sum = 0.0;
    for (size_t i = 0; i < vals.size(); ++i)
    {
        sum += vals.at(i);
    }
    return sum / ((float)vals.size());
}

void RunOnTestSet(
    LinearLayer& a_firstLayer,
    ReLULayer& a_secondLayer,
    LinearLayer& a_thirdLayer,
    SoftmaxLayer& a_sofmaxLayer,
    MNISTDataloader& a_testDataloader,
    size_t a_batchSize,
    vector<float> a_confidenceCutoffs)
{
    LOG(INFO) << "Processing Test Set..." << endl;
    // instantiate the metrics we want to see
    vector<shared_ptr<metrics::Metric>> l_metrics = {
        shared_ptr<metrics::Metric>(new metrics::Precision()),
        shared_ptr<metrics::Metric>(new metrics::Recall())
    };

    size_t totalIters = a_testDataloader.GetNumBatches(a_batchSize);
    for (size_t i = 0; i < totalIters; ++i)
    {
        TMutableTensorPtr l_inputs, l_targets;
        a_testDataloader.GetNextBatch(l_inputs, l_targets, a_batchSize);

        // Forward pass to probs
        TTensorPtr l_output0 = a_firstLayer.Forward(l_inputs);
        TTensorPtr l_output1 = a_secondLayer.Forward(l_output0);
        TTensorPtr l_output2 = a_thirdLayer.Forward(l_output1);
        TTensorPtr l_probs = a_sofmaxLayer.Forward(l_output2);

        // Accumulate metrics
        for (auto& metric : l_metrics) {
            metric->AddResults(l_probs, l_targets);
        }
    }

    // Print precision recall curve for test set
    for (int i = 0; i < a_confidenceCutoffs.size(); ++i)
    {
        for (const auto& metric : l_metrics) {
            float val = metric->Calculate(a_confidenceCutoffs.at(i)) * 100.0;
            LOG(INFO) << "Test " << metric->GetName() << " @" 
                      << a_confidenceCutoffs.at(i) << " = "
                      << val << "%" << endl;
        }
    }
}

int main(int argc, char const *argv[])
{
    // Define data loader
    string l_dataPath = "../data/mnist/";
    MNISTDataloader l_trainDataloader(l_dataPath, true); // second param for isTrain?
    MNISTDataloader l_testDataloader(l_dataPath, false); // second param for isTrain?

    // Define model
    // first linear layer is 784x300
    // 784 inputs, 300 hidden size
    LinearLayer firstLinearLayer(Tensor::Random({784, 300}, -0.01f, 0.01f));

    // Non-linear activation
    ReLULayer activationLayer;
    
    // second linear layer is 300x10
    // 300 hidden units, 10 outputs
    LinearLayer secondLinearLayer(Tensor::Random({300, 10}, -0.01f, 0.01f));

    // Convert outputs to probabilities
    SoftmaxLayer softmaxLayer;

    // Error function
    CrossEntropyLoss loss;

    // Training loop
    float learningRate = 0.0001;
    size_t numEpochs = 1000;
    size_t batchSize = 100;
    float lastTestAcc = 0.0;

    size_t totalIters = l_trainDataloader.GetNumBatches(batchSize);
    for (size_t i = 0; i < numEpochs; ++i)
    {
        LOG(INFO) << "====== BEGIN EPOCH " << i << " ======" << endl;
        metrics::Accuracy l_accuracyMetric;
        vector<float> errorAcc;
        for (size_t j = 0; j < totalIters; ++j)
        {
            // Get training example
            TMutableTensorPtr input, target;
            l_trainDataloader.GetNextBatch(input, target, batchSize);

            // Forward pass
            TTensorPtr output0 = firstLinearLayer.Forward(input);
            TTensorPtr output1 = activationLayer.Forward(output0);
            TTensorPtr output2 = secondLinearLayer.Forward(output1);
            TTensorPtr probs = softmaxLayer.Forward(output2);

            // Accumulate accuracy
            l_accuracyMetric.AddResults(probs, target);

            // Calc Error
            float error = loss.Forward(probs, target);
            errorAcc.push_back(error);

            // Backward pass
            TTensorPtr errorGrad = loss.Backward(probs, target);
            TTensorPtr probsGrad = softmaxLayer.Backward(output2, errorGrad);
            TTensorPtr grad2 = secondLinearLayer.Backward(output1, probsGrad);
            TTensorPtr grad1 = activationLayer.Backward(output0, grad2);
            TTensorPtr grad0 = firstLinearLayer.Backward(input, grad1);

            // Gradient Descent
            secondLinearLayer.UpdateWeights(learningRate);
            firstLinearLayer.UpdateWeights(learningRate);

            // Only log every 100 examples
            
            if (j % 100 == 0)
            {
                float avgError = CalcAverage(errorAcc);
                float accuracy = l_accuracyMetric.Calculate() * 100;
                LOG(INFO) << "--ITER (" << i << "," << j << "/" << totalIters 
                          << ")-- avgError = " << avgError 
                          << " avgAcc = " << accuracy 
                          << " lr = " << learningRate << endl;
                for (size_t k = 0; k < probs->Shape().at(1); ++k)
                {
                    LOG(INFO) << "Output [" << k << "] " << probs->At({0, k}) << " Target " << target->At({0, k}) << endl;
                }

                size_t targetVal = target->GetRow(0)->MaxIdx();
                size_t predVal = probs->GetRow(0)->MaxIdx();

                LOG(INFO) << "Got prediction: " << predVal << " for target " << targetVal << endl;
            }
        }

        // Calc tran accuracy
        float accuracy = l_accuracyMetric.Calculate() * 100;
        LOG(INFO) << "Train accuracy (" << i << ") " << " = " << accuracy
                  << "%" << " last test acc: " << lastTestAcc << endl;
        
        // Calculate test set precision / recall curve at confidences
        vector<float> l_confidences = {0.1, 0.15, 0.25, 0.5, 0.75, 0.9};
        RunOnTestSet(
            firstLinearLayer,
            activationLayer,
            secondLinearLayer,
            softmaxLayer,
            l_testDataloader,
            batchSize,
            l_confidences);

        if (i % 50 == 0 && i > 0)
        {
            learningRate *= 0.75;
        }
        
        lastTestAcc = accuracy;
        errorAcc.clear();
    }

    return 0;
}
