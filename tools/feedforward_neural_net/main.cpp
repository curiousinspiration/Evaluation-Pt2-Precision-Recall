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

#include <math.h>

#include <glog/logging.h>

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

float CalcAccuracy(
    LinearLayer& a_firstLayer,
    ReLULayer& a_secondLayer,
    LinearLayer& a_thirdLayer,
    SoftmaxLayer& a_sofmaxLayer,
    MNISTDataloader& a_testDataloader,
    size_t a_batchSize)
{
    float numCorrect = 0.0;
    float numTotal = 0.0;
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
            size_t targetVal = target->GetRow(j)->MaxIdx();
            size_t predVal = probs->GetRow(j)->MaxIdx();
            if (predVal == targetVal)
            {
                ++numCorrect;
            }
            ++numTotal;
        }

        if (i % 100 == 0)
        {
            LOG(INFO) << "Processing test set... " << i << endl;
        }
    }

    float l_accuracy = (numCorrect / numTotal) * 100;
    LOG(INFO) << "Test accuracy = " << numCorrect << "/" << numTotal 
              << " = " << l_accuracy << "%" << endl;
    return l_accuracy;
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
    float learningRate = 0.0003;
    size_t numEpochs = 100;
    size_t batchSize = 8;
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
                    LOG(INFO) << "Output: " << probs->At({0, k}) << " Target " << target->At({0, k}) << endl;
                }

                size_t targetVal = target->GetRow(0)->MaxIdx();
                size_t predVal = probs->GetRow(0)->MaxIdx();

                LOG(INFO) << "Got prediction: " << predVal << " for target " << targetVal << endl;
                errorAcc.clear();
            }
        }
        float accuracy = ((float) numCorrect / (float) numTotal) * 100;
        LOG(INFO) << "Train accuracy (" << i << ") " << numCorrect  << " / " << numTotal << " = " << accuracy << "%" << " last test acc: " << lastTestAcc << endl;
        float testAccuracy = CalcAccuracy(firstLinearLayer, activationLayer, secondLinearLayer, softmaxLayer, l_testDataloader, batchSize);

        if (i < 5 || (lastTestAcc+1.0 < testAccuracy))
        {
            learningRate /= 2.0;
        }
        
        lastTestAcc = accuracy;
    }

    return 0;
}
