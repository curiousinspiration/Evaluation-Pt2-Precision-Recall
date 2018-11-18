/*
 * ADataloader Implementation
 */

#include "neural/data/dataloader.h"
#include "neural/math/math.h"

#include <glog/logging.h>

#include <stdexcept>
#include <algorithm>

using namespace std;

namespace neural
{

ADataloader::ADataloader(bool a_shouldRandomize)
    : m_shouldRandomize(a_shouldRandomize)
    , m_numData(0)
    , m_currentIdx(0)
{
}

void ADataloader::GetNextBatch(
    TMutableTensorPtr& a_outInput,
    TMutableTensorPtr& a_outOutput,
    size_t a_batchSize)
{
    // initialize indices (cant call pure virtual methods in constructor)
    if (m_indices.empty())
    {
        LOG(INFO) << "Dataloader creating data indices..." << endl;
        m_numData = DataLength();
        for (size_t i = 0; i < m_numData; ++i)
        {
            m_indices.push_back(i);
        }

        LOG(INFO) << "Dataloader randomizing data indices..." << endl;
        if (m_shouldRandomize)
        {
            std::random_shuffle(m_indices.begin(), m_indices.end());
        }
    }

    LOG(INFO) << "Dataloader getting batch of size " << a_batchSize << endl;
    if (m_currentIdx >= m_numData)
    {
        LOG(INFO) << "Dataloader randomizing data indices..." << endl;
        std::random_shuffle(m_indices.begin(), m_indices.end());
        m_currentIdx = 0;
    }

    size_t l_dataIdx = m_indices.at(m_currentIdx);
    ++m_currentIdx;

    TMutableTensorPtr l_input, l_output;
    // Populate data at index
    DataAt(l_dataIdx, l_input, l_output);

    a_outInput = Tensor::New({a_batchSize, l_input->Shape().at(1)});
    a_outOutput = Tensor::New({a_batchSize, l_output->Shape().at(1)});

    a_outInput->SetRow(0, l_input);
    a_outOutput->SetRow(0, l_output);

    for (int i = 1; i < a_batchSize; ++i)
    {
        if (m_currentIdx >= m_numData)
        {
            LOG(INFO) << "Dataloader randomizing data indices..." << endl;
            std::random_shuffle(m_indices.begin(), m_indices.end());
            m_currentIdx = 0;
        }

        size_t l_dataIdx = m_indices.at(m_currentIdx);
        TMutableTensorPtr l_input, l_output;
        // Populate data at index
        DataAt(l_dataIdx, l_input, l_output);
        a_outInput->SetRow(i, l_input);
        a_outOutput->SetRow(i, l_output);
        ++m_currentIdx;
    }

    /*
    bool l_haveInitializedIO = false;
    for (int i = 0; i < a_batchSize; ++i)
    {
        if (m_currentIdx >= m_numData)
        {
            std::random_shuffle(m_indices.begin(), m_indices.end());
            m_currentIdx = 0;
        }

        size_t l_dataIdx = m_indices.at(m_currentIdx);

        TMutableTensorPtr l_input, l_output;
        // Populate data at index
        DataAt(l_dataIdx, l_input, l_output);

        if (!l_haveInitializedIO)
        {
            a_outInput = l_input;
            a_outOutput = l_output;
            l_haveInitializedIO = true;
            LOG(INFO) << "Got first input: " << a_outInput->ShapeStr()
                      << " output: " << a_outOutput->ShapeStr() << endl;
        }
        else
        {
            a_outInput = Math::CatMutable(a_outInput, l_input);
            a_outOutput = Math::CatMutable(a_outOutput, l_output);
            LOG(INFO) << "Concatenated input: " << a_outInput->ShapeStr()
                      << " output: " << a_outOutput->ShapeStr() << endl;
        }

        ++m_currentIdx;
    }
    */
    LOG(INFO) << "GOT BATCH input: " << a_outInput->ShapeStr()
              << " output: " << a_outOutput->ShapeStr() << endl;
}

size_t ADataloader::GetNumBatches(size_t a_batchSize) const
{
    return DataLength() / a_batchSize;
}

} // namespace neural
