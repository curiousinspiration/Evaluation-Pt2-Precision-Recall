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

bool p_RowIsTruePositive(
    size_t a_targetIdx, size_t a_predIdx,
    float a_guessConfidence, float a_confidenceCutoff)
{
    return (a_targetIdx == a_predIdx) and (a_guessConfidence > a_confidenceCutoff);
}

bool p_RowIsFalsePositive(
    size_t a_targetIdx, size_t a_predIdx,
    float a_guessConfidence, float a_confidenceCutoff)
{
    return (a_targetIdx != a_predIdx) and (a_guessConfidence > a_confidenceCutoff);
}

bool p_RowIsTrueNegative(
    size_t a_targetIdx, size_t a_predIdx,
    float a_guessConfidence, float a_confidenceCutoff)
{
    return (a_targetIdx != a_predIdx) and (a_guessConfidence < a_confidenceCutoff);
}

bool p_RowIsFalseNegative(
    size_t a_targetIdx, size_t a_predIdx,
    float a_guessConfidence, float a_confidenceCutoff)
{
    return (a_targetIdx == a_predIdx) and (a_guessConfidence < a_confidenceCutoff);
}

map<string, float> CalcStatsAtConfidence(
    LinearLayer& a_firstLayer,
    ReLULayer& a_secondLayer,
    LinearLayer& a_thirdLayer,
    SoftmaxLayer& a_sofmaxLayer,
    MNISTDataloader& a_testDataloader,
    size_t a_batchSize,
    vector<float> a_confidenceCutoffs)
{
    LOG(INFO) << "Processing Test Set..." << endl;
    vector<float> l_truePositives;
    vector<float> l_trueNegatives;
    vector<float> l_falsePositives;
    vector<float> l_falseNegatives;
    for (size_t i = 0; i < a_confidenceCutoffs.size(); ++i)
    {
        l_truePositives.push_back(0.0);
        l_trueNegatives.push_back(0.0);
        l_falsePositives.push_back(0.0);
        l_falseNegatives.push_back(0.0);
    }

    size_t totalIters = a_testDataloader.GetNumBatches(a_batchSize);
    for (size_t i = 0; i < totalIters; ++i)
    {
        TMutableTensorPtr input, target;
        a_testDataloader.GetNextBatch(input, target, a_batchSize);

        // Forward pass
        TTensorPtr l_output0 = a_firstLayer.Forward(input);
        TTensorPtr l_output1 = a_secondLayer.Forward(l_output0);
        TTensorPtr l_output2 = a_thirdLayer.Forward(l_output1);
        TTensorPtr probs = a_sofmaxLayer.Forward(l_output2);

        for (size_t j = 0; j < a_batchSize; ++j)
        {
            size_t l_targetIdx = target->GetRow(j)->MaxIdx();
            size_t l_predIdx = probs->GetRow(j)->MaxIdx();
            float l_predVal = probs->GetRow(j)->MaxVal();

            for (int k = 0; k < a_confidenceCutoffs.size(); ++k)
            {
                if (p_RowIsTruePositive(l_targetIdx, l_predIdx, l_predVal, a_confidenceCutoffs.at(k)))
                {
                    l_truePositives.at(k) += 1.0;
                }

                if (p_RowIsTrueNegative(l_targetIdx, l_predIdx, l_predVal, a_confidenceCutoffs.at(k)))
                {
                    l_trueNegatives.at(k) += 1.0;
                }

                if (p_RowIsFalsePositive(l_targetIdx, l_predIdx, l_predVal, a_confidenceCutoffs.at(k)))
                {
                    l_falsePositives.at(k) += 1.0;
                }

                if (p_RowIsFalseNegative(l_targetIdx, l_predIdx, l_predVal, a_confidenceCutoffs.at(k)))
                {
                    l_falseNegatives.at(k) += 1.0;
                }
            }
        }
    }

    for (int i = 0; i < a_confidenceCutoffs.size(); ++i)
    {
        float l_accuracy = (l_truePositives.at(i) + l_trueNegatives.at(i)) / (l_truePositives.at(i) + l_trueNegatives.at(i) + l_falsePositives.at(i) + l_falseNegatives.at(i));
        l_accuracy *= 100;
        float l_precision = (l_truePositives.at(i)) / (l_truePositives.at(i) + l_falsePositives.at(i));
        l_precision *= 100;
        float l_recall = (l_truePositives.at(i)) / (l_truePositives.at(i) + l_falseNegatives.at(i));
        l_recall *= 100;
        LOG(INFO) << "Test accuracy @" << a_confidenceCutoffs.at(i) << " = " << l_accuracy << "%" << endl;
        LOG(INFO) << "Test precision @" 
                  << a_confidenceCutoffs.at(i)
                  << " = "
                  << l_precision << "% = " 
                  << l_truePositives.at(i) << " / (" 
                  << l_truePositives.at(i) << " + " 
                  << l_falsePositives.at(i) << ")"
                  << endl;
        LOG(INFO) << "Test recall @" 
                  << a_confidenceCutoffs.at(i) 
                  << " = " 
                  << l_recall << "% = "
                  << l_truePositives.at(i) << " / (" 
                  << l_truePositives.at(i) << " + " 
                  << l_falseNegatives.at(i) << ")"
                  << endl;
    }
    
    return map<string, float>({{"accuracy", 0.0}, {"precision", 0.0}, {"recall", 0.0}});
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
    // MeanSquaredErrorLoss loss;
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
        size_t numCorrect = 0;
        size_t numTotal = 0;
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

            // Count num correct in batch
            for (size_t k = 0; k < batchSize; ++k)
            {
                size_t targetVal = target->GetRow(k)->MaxIdx();
                size_t predVal = probs->GetRow(k)->MaxIdx();
                if (predVal == targetVal)
                {
                    ++numCorrect;
                }
                ++numTotal;
            }

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
                float accuracy = ((float) numCorrect / (float) numTotal) * 100;

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
                errorAcc.clear();
            }
        }
        float accuracy = ((float) numCorrect / (float) numTotal) * 100;
        LOG(INFO) << "Train accuracy (" << i << ") " << numCorrect  << " / " << numTotal << " = " << accuracy << "%" << " last test acc: " << lastTestAcc << endl;
        
        vector<float> l_confidences = {0.1, 0.15, 0.25, 0.5, 0.75, 0.9};
        CalcStatsAtConfidence(firstLinearLayer, activationLayer, secondLinearLayer, softmaxLayer, l_testDataloader, batchSize, l_confidences);

        if (i % 50 == 0 && i > 0)
        {
            learningRate *= 0.75;
        }
        
        lastTestAcc = accuracy;
    }

    return 0;
}
