/*
 * Dataloader Implementation
 */

#include "neural/data/dataloader.h"

#include <glog/logging.h>

#include <stdexcept>
#include <algorithm>

using namespace std;

namespace neural
{

Dataloader::Dataloader(bool a_shouldRandomize)
    : m_shouldRandomize(a_shouldRandomize)
    , m_numData(0)
    , m_currentIdx(0)
{
}

void Dataloader::GetNextBatch(
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

    // If our current index pointer is > data length, reshuffle and set to 0
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
        // could go over m_numData here
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
}

size_t Dataloader::GetNumBatches(size_t a_batchSize) const
{
    return DataLength() / a_batchSize;
}

} // namespace neural
