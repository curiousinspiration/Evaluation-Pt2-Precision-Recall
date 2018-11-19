/*
 * Dataloader
 */

#pragma once

#include "neural/math/tensor.h"

#include <memory>

namespace neural
{

class Dataloader
{
public:
    Dataloader(bool a_shouldRandomize = true);

    // Implement these functions for your dataloader
    virtual size_t DataLength() const = 0;
    virtual bool DataAt(
        size_t i,
        TMutableTensorPtr& a_outInput,
        TMutableTensorPtr& a_outOutput) const = 0;

    void GetNextBatch(
        TMutableTensorPtr& a_outInput,
        TMutableTensorPtr& a_outOutput,
        size_t a_batchSize);
    size_t GetNumBatches(size_t a_batchSize) const;
private:
    bool m_shouldRandomize;
    size_t m_numData;
    size_t m_currentIdx;
    std::vector<size_t> m_indices;

};

} // namespace neural
