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
    template <typename T>
    struct LayerParameters {
        size_t neurons; // Number of neurons
        size_t weights = 0; // 0 if fully connected to previous layer
        T min_weight = -2;
        T max_weight = +2;
    };

    template <typename T>
    class Net {
    public:
        std::vector< std::vector<T> > neuron_layers;
        std::vector< std::vector<T> > weight_layers;
        std::vector< std::vector<T> > weight_changes;

        void
        init(const std::vector< LayerParameters<T> > & parameters, int seed = 0);

        Net(const std::vector< LayerParameters<T> > & parameters, int seed = 0);
        Net();
        void feed_forward(std::vector<T> & from,
                          std::vector<T> & to,
                          std::vector<T> & weights);
        std::vector<T> output(const std::vector<T> & pattern);
        void train_single(const std::vector<T> & pattern,
                          const std::vector<T> & target,
                          float eta, float momentum);
        void train_online(const std::vector< std::vector<T> > & patterns,
                          const std::vector< std::vector<T> > & targets,
                          float eta, float momentum, unsigned int iterations);
        void train_batch(const std::vector< std::vector<T> > & patterns,
                         const std::vector< std::vector<T> > & targets,
                         float eta,
                         float momentum,
                         unsigned int iterations);
        void train(const std::vector< std::vector<T> > & patterns,
                   const std::vector< std::vector<T> > & targets,
                   float eta,
                   float momentum,
                   unsigned int iterations,
                   bool online=true);
        double mean_squared_error(const std::vector< std::vector<T> > & patterns,
                                  const std::vector< std::vector<T> > & targets);
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
