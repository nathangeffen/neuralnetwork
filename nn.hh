/*
 * @file nn.hh Declares, and defines a few of, the core neural network
 * functions, and the nn namespace.
 *
 */

#ifndef NN_HH
#define NN_HH

#include <algorithm>
#include <cassert>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <ctime>
#include <utility>
#include <stdexcept>

namespace nn {

    /**
     * This structure is used to define each layer of a net.
     */

    template <typename T>
    struct LayerParameters {
        size_t neurons; ///< Number of neurons
        size_t weights = 0; ///< 0 if fully connected to previous layer
        T min_weight = -2; ///< minimum weight to initialize weight with
        T max_weight = +2; ///< maximum weight to initialize weight with
    };

    /**
     * This is the neural network class. The T parameter is in case this is
     * executed on hardware for which the natural type is not a four-byte float.
     */

    template <typename T>
    class Net {
    public:
        /// Container for the neurons
        std::vector< std::vector<T> > neuron_layers;
        /// Container for the weights (usually one layer less than neurons)
        std::vector< std::vector<T> > weight_layers;
        /// Used by training algorithms to store previous weight changes
        std::vector< std::vector<T> > weight_changes;

        void
        init(const std::vector< LayerParameters<T> > & parameters, int seed = 0);

        Net(const std::vector< LayerParameters<T> > & parameters, int seed = 0);
        Net();
        void feed_forward(const std::vector<T> & from,
                          std::vector<T> & to,
                          const std::vector<T> & weights);
        std::vector<T> output(typename std::vector<T>::const_iterator first,
                              typename std::vector<T>::const_iterator last);
        std::vector<T> output(const std::vector<T> & pattern);
        void train_single(typename std::vector<T>::const_iterator pattern_first,
                          typename std::vector<T>::const_iterator pattern_last,
                          typename std::vector<T>::const_iterator target_first,
                          typename std::vector<T>::const_iterator target_last,
                          float eta, float momentum);
        void train_single(const std::vector<T> & pattern,
                          const std::vector<T> & target,
                          float eta, float momentum);
        void train_online(
            typename std::vector< std::vector<T> >::const_iterator patterns_first,
            typename std::vector< std::vector<T> >::const_iterator patterns_last,
            typename std::vector< std::vector<T> >::const_iterator targets_first,
            float eta, float momentum, unsigned int iterations);
        void train_online(const std::vector< std::vector<T> > & patterns,
                          const std::vector< std::vector<T> > & targets,
                          float eta, float momentum, unsigned int iterations);
        void train_batch(
            typename std::vector< std::vector<T> >::const_iterator patterns_first,
            typename std::vector< std::vector<T> >::const_iterator patterns_last,
            typename std::vector< std::vector<T> >::const_iterator targets_first,
            float eta,
            unsigned int iterations);
        void train_batch(const std::vector< std::vector<T> > & patterns,
                         const std::vector< std::vector<T> > & targets,
                         float eta,
                         unsigned int iterations);
        void train(
            typename std::vector< std::vector<T> >::const_iterator patterns_first,
            typename std::vector< std::vector<T> >::const_iterator patterns_last,
            typename std::vector< std::vector<T> >::const_iterator targets_first,
            float eta,
            float momentum,
            unsigned int iterations,
            bool online=true);
        void train(const std::vector< std::vector<T> > & patterns,
                   const std::vector< std::vector<T> > & targets,
                   float eta,
                   float momentum,
                   unsigned int iterations,
                   bool online=true);
        double mean_squared_error(
            typename std::vector< std::vector<T> >::const_iterator patterns_first,
            typename std::vector< std::vector<T> >::const_iterator patterns_last,
            typename std::vector< std::vector<T> >::const_iterator targets_first);
        double mean_squared_error(const std::vector< std::vector<T> > & patterns,
                                  const std::vector< std::vector<T> > & targets);
        unsigned int score(
            typename std::vector< std::vector<T> >::const_iterator patterns_first,
            typename std::vector< std::vector<T> >::const_iterator patterns_last,
            typename std::vector< std::vector<T> >::const_iterator targets_first);
        unsigned int score(const std::vector< std::vector<T> > & patterns,
                           const std::vector< std::vector<T> > & targets);
        void print(bool print_weights=true, bool print_neurons=false);
        void save(const char *filename);
    private:
        std::mt19937 rng_;
    };

    template <typename T>
    T sigmoid(T f);

    template <typename T>
    void sigmoid_vector(std::vector<T> & vec);

    template <typename T> void
    print_vector(const std::vector<T> & vec,
                 const std::string & delim = "\n");

    template <typename T> void
    print_matrix(const std::vector< std::vector<T> > & matrix);

    template <typename T>
    Net<T> load(const char *filename);
}

#endif
