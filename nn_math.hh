/*
 * @file nn_math.hh Declares useful math functions for neural networks.
 *
 */

#ifndef NN_MATH_HH
#define NN_MATH_HH

#include <algorithm>
#include <functional>
#include <numeric>
#include <cmath>

namespace nn {

    /**
     * The iconic multilayered pereceptron firing function
     *
     * @param f The sum of the inputs to a neuron
     */
    template <typename T>
    T sigmoid(T f) {
        return (T) 1.0 / (1.0 + exp(-f));
    }

    /**
     * Calculates the mean of a dataset.
     *
     * @param first Iterator pointing to first element
     *
     * @param last  Iterator pointing to last element
     */
    template <typename T=float, typename IteratorType>
    T mean(IteratorType first, IteratorType last) {
        return std::accumulate(first, last, (T) 0, std::plus<T>()) /
            std::distance(first, last);
    }

    /**
     * Calculates the standard deviation of a dataset
     *
     * @param first Iterator pointing to first element
     *
     * @param last  Iterator pointing to last element
     *
     * @param mean The mean of the dataset. If this is unknown
     * use the version of this function without this parameter.
     */
    template <typename T=float, typename IteratorType>
    T stdev(IteratorType first, IteratorType last, T mean) {
        T total = (T) 0;
        for (auto it = first; it != last; it++) {
            auto t = (*it - mean);
            total += t * t;
        }
        return std::sqrt(total/(std::distance(first, last) - 1));
    }

    /**
     * Calculates the standard deviation of a dataset
     *
     * @param first Iterator pointing to first element
     *
     * @param last  Iterator pointing to last element
     *
     */
    template <typename T=float, typename IteratorType>
    T stdev(IteratorType first, IteratorType last) {
        T m = mean<T>(first, last);
        return stdev<T>(first, last, m);
    }

    template <typename T, typename IteratorType>
    struct NormalizeResult {
        T min;
        T max;
        IteratorType first;
    };

    /**
     * Normalizes a list or array of numbers so that they are scaled to fit
     * between a low and high number. Call this on the non-training input data of a
     * neural net.
     *
     * @param first Iterator pointing to first element
     *
     * @param last  Iterator pointing to last element
     *
     * @param lo The low end of the range to which the elements are scaled
     *
     * @param hi The high end of the range to which the elements are scaled
     *
     * @param x_min The lowest number for this feature in the training set
     *
     * @param x_max The highest number for this feature in the training set
     *
     */
    template <typename T=float, typename IteratorType>
    IteratorType normalize(IteratorType first, IteratorType last,
                           T lo, T hi, T x_min, T x_max) {
        const T max_min = x_max - x_min;
        const T hi_lo = hi - lo;
        for (auto it = first; it != last; it++) {
            T x = (T) *it;
            *it = ( (x - x_min) / max_min ) * hi_lo + lo;
        }
        return first;
    }

    /**
     * Normalizes a list or array of numbers so that they are scaled to fit
     * between a low and high number. Call this on the training input data of a
     * neural net.
     *
     * @param first Iterator pointing to first element
     *
     * @param last  Iterator pointing to last element
     *
     * @param lo The low end of the range to which the elements are scaled
     *
     * @param hi The high end of the range to which the elements are scaled
     *
     */
    template <typename T=float, typename IteratorType>
    NormalizeResult<T, IteratorType>
    normalize(IteratorType first, IteratorType last, T lo, T hi) {
        NormalizeResult<T, IteratorType> result;
        const T x_min = *std::min_element(first, last);
        const T x_max = *std::max_element(first, last);
        normalize(first, last, lo, hi, x_min, x_max);

        result.min = x_min;
        result.max = x_max;
        result.first = first;
        return result;
    }

}

#endif
