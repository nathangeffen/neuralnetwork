/*
 * @file nn_preprocess.hh Preprocesses patterns for neural net.
 *
 */

#ifndef NN_PREPROCESS_HH
#define NN_PREPROCESS_HH

#include "nn_math.hh"
#include "nn_matrix.hh"

namespace nn {

    template <typename T=float, typename IteratorType>
    Matrix<T> patterns_to_matrix(IteratorType first, IteratorType last) {
        return Matrix<float>(last - first,
                             (*first).size(),
                             [&](T& val, unsigned row, unsigned col) {
                                 val = (*(first + row))[col];
                             });
    }
};

#endif
