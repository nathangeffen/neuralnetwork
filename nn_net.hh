/*
 * @file nn.hh Declares, and defines a few of, the core neural network
 * functions, and the nn namespace.
 *
 */

#ifndef NN_NET_HH
#define NN_NET_HH

#include <algorithm>
#include <cassert>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <ctime>
#include <utility>
#include <stdexcept>
#include "nn_math.hh"
#include "nn_matrix.hh"
#include "nn_csv.hh"

namespace nn {

    /**
     * Executes sigmoid on a bunch of neurons
     * @param vec Usually a layer of neurons that have been summed
     */
    template <typename T>
    void sigmoid_vector(std::vector<T> & vec) {
        size_t i;
        const auto n = vec.size();
        for (i = 0; i < n; i++) {
            vec[i] = sigmoid<T>(vec[i]);
        }
    }

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

        /**
         * Default constructor. Does nothing.
         */
        Net() {}

        /**
         * There should be a set of parameters for each layer.
         * The input layer parameters are only used to determine number of neurons.
         * i.e. The number of weights etc in the input layer are ignored.
         *
         * @param parameters Specification for each layer of the net
         *
         * @param seed Used to seed the random generator (0 for random device to be
         * used)
         */
        Net(const std::vector< LayerParameters<T> > & parameters, int seed = 0) {
            init(parameters, seed);
        }

        /**
         * Calculates the net's output for a given input pattern
         *
         * @param first The first entry of the input pattern
         * @param last The last entry of the input pattern
         *
         * @return The output layer of the neural net
         */
        template <typename Iterator>
        std::vector<T> calc_output(Iterator first,
                                   Iterator last) {
            assert( (size_t) (last - first) == neuron_layers[0].size());
            std::copy(first, last, neuron_layers[0].begin());
            for (size_t i = 1; i < neuron_layers.size(); i++) {
                feed_forward(neuron_layers[i-1], neuron_layers[i],
                             weight_layers[i-1]);
                sigmoid_vector(neuron_layers[i]);
            }
            return neuron_layers.back();
        }


        /**
         * Calculates the net's output for a given input pattern
         *
         * @param pattern The pattern for which to calculate the output
         *
         * @return The output layer of the neural net
         */
        std::vector<T> calc_output(const std::vector<T> & pattern) {
            return calc_output(pattern.begin(), pattern.end());
        }

        /**
         * Executes the backpropagation algorithm on a single training-target pair.
         * The result is stored in the last (output) layer of the net.
         *
         * @param pattern_first Iterator pointing to first element of pattern
         * @param pattern_last Iterator pointing to last element of pattern
         * @param target_first Iterator pointing to first element of target
         * @param target_last Iterator pointing to last element of target
         * @param eta Factor by which to multiply the weight update deltas
         * @param momentum Factor by which to multiply the previous weight change
         *                 on the new update
         *
         */
        template <typename Iterator>
        void train_single(Iterator pattern_first,
                          Iterator pattern_last,
                          Iterator target_first,
                          Iterator target_last,
                          float eta, float momentum) {
            size_t i, j, k, l;
            T change;
            assert(neuron_layers.size() > 2);
            assert( (size_t) (pattern_last - pattern_first) ==
                    neuron_layers.front().size());
            assert( (size_t) (target_last - target_first) ==
                    neuron_layers.back().size());

            // Calculate the errors: difference between target and pattern
            calc_output(pattern_first, pattern_last);
            std::vector<T> curr_deltas(neuron_layers.back().size());

            for (auto it_d = curr_deltas.begin(), it_t = target_first,
                     it_n = neuron_layers.back().begin();
                 it_t != target_last;
                 it_d++, it_t++, it_n++) {
                *it_d = *it_t - *it_n;
            }

            // Step through the layers from output to first hidden layer
            for (l = neuron_layers.size() - 1; l > 0; l--) {
                // curr is the neuron layer we are working with
                auto & curr = neuron_layers[l];
                // prev is the neuron layer feeding curr
                auto & prev = neuron_layers[l - 1];
                // weights are the connecrtions between prev and curr
                auto & weights = weight_layers[l - 1];
                // changes are the last recorded change for each weight
                auto & changes = weight_changes[l - 1];
                // prev_deltas is used to calculate the sum of deltas for the prev
                // layer
                std::vector<T> prev_deltas(prev.size(), 0);

                // We step through each neuron in the current layer
                for (i = 0; i < curr.size(); i++) {
                    // Derivative of the error
                    T deriv = curr[i] * (1.0 - curr[i]);
                    // The backpropagated sum of errors times the derivative
                    T delta = curr_deltas[i] * deriv;
                    // We step through the neurons of the previous layer
                    auto n = curr.size();
                    #pragma omp parallel for
                    for (j = 0; j < prev.size(); j++) {
                        // This is the index into the weight connections
                        // between the previous layer and the current layer.
                        // Currently this presumes fully connected.
                        k = j * n + i;
                        // Add the error
                        prev_deltas[j] += weights[k] * delta;
                        // The total change to the weight excluding momentum
                        change = delta * eta * prev[j];
                        // Update the weight
                        weights[k] += change + momentum * changes[k];
                        // save the change excluding momentum for the next
                        // update
                        changes[k] = change;
                    }
                    // Now update the bias connection to neuron[i]
                    // Get index to the correct connection to update
                    k = prev.size() * curr.size() + i;
                    // Calculate the change for the (negative) bias neuron
                    change = -delta * eta;
                    // Update the weight
                    weights[k] += change + momentum * changes[k];
                    // Save the change excluding momentum for the next
                    // update
                    changes[k] = change;
                }
                // Replace deltas of current layer with deltas of previous one
                curr_deltas = prev_deltas;
            }

        }

        /**
         * Executes the backpropagation algorithm on a single training-target
         * pair. The result is stored in the last (output) layer of the net.
         *
         * @param pattern Pattern on which to train the net
         * @param target Expected output for the pattern
         * @param eta Factor by which to multiply the weight update deltas
         * @param momentum Factor by which to multiply the previous weight change
         *                 on the new update
         *
         */
        template <typename Container>
        void train_single(const Container & pattern,
                          const Container & target,
                          float eta, float momentum) {
            train_single(pattern.begin(), pattern.end(),
                         target.begin(), target.end(), eta, momentum);
        }

        /**
         * Uses so-called online training method on a set of patterns and targets,
         * in which weights are updated after each pattern is presented.
         *
         * @param patterns_first Iterator pointing to first pattern in set
         *
         * @param patterns_last Iterator pointing beyond the last pattern in set
         *
         * @param targets_first Iterator pointing to the first target (i.e. the
         * first output vector that the net must be trained to match on the
         * corresponding pattern) in a set. There must be an equal number of
         * patterns and targets.
         *
         * @param eta Factor by which to multiply the weight update deltas
         *
         * @param momentum Factor by which to multiply the previous weight change
         *                 on the new update
         *
         * @param iterations Number of iterations to train for
         *
         */
        template <typename Iterator>
        void train_online(Iterator patterns_first,
                          Iterator patterns_last,
                          Iterator targets_first,
                          float eta, float momentum, unsigned int iterations) {
            size_t n = patterns_last - patterns_first;
            std::vector<size_t>  indices(n);
            for (size_t i = 0; i < indices.size(); i++) indices[i] = i;
            for (unsigned int i = 0; i < iterations; i++) {
                std::shuffle(indices.begin(), indices.end(), rng_);
                for (auto index: indices) {
                    train_single(*(patterns_first + index),
                                 *(targets_first + index), eta, momentum);
                }
            }
        }

        /**
         * Uses so-called online training method on a set of patterns and
         * targets, in which weights are updated after each pattern is presented.
         *
         * @param patterns Set of patterns to train with
         *
         * @param targets Set of targets for corresponding patterns that net must be
         * trained to match.
         *
         * @param eta Factor by which to multiply the weight update deltas
         *
         * @param momentum Factor by which to multiply the previous weight change
         *                 on the new update
         *
         * @param iterations Number of iterations to train for
         *
         */
        template <typename Container>
        void train_online(const Container & patterns,
                          const Container & targets,
                          float eta, float momentum, unsigned int iterations) {
            train_online(patterns.begin(), patterns.end(), targets.begin(),
                         eta, momentum, iterations);
        }

        /**
         * Uses so-called batch training method on a set of patterns and
         * targets, in which weights are updated after all the patterns are
         * presented. The method works on fully connected nets using the sigmoid
         * function.
         *
         * @param patterns_first Iterator pointing to first pattern in set
         *
         * @param patterns_last Iterator pointing beyond the last pattern in set
         *
         * @param targets_first Iterator pointing to the first target (i.e. the
         * first output vector that the net must be trained to match on the
         * corresponding pattern) in a set. There must be an equal number of
         * patterns and targets.
         *
         * @param eta Factor by which to multiply the weight update deltas
         *
         * @param iterations Number of iterations to train for
         *
         */
        template <typename Iterator>
        void train_batch(Iterator patterns_first,
                         Iterator patterns_last,
                         Iterator targets_first,
                         float eta,
                         unsigned int iterations) {
            size_t i, j, k, l;
            unsigned int c;
            assert(neuron_layers.size() > 2);
            assert(patterns_last - patterns_first > 0);
            assert(patterns_first->size() == neuron_layers.front().size());
            assert(neuron_layers.back().size() == targets_first->size());

            for (c = 0; c < iterations; c++) {
                std::vector< std::vector<T> > grads;
                for (auto it = neuron_layers.begin() + 1;
                     it < neuron_layers.end(); it++) {
                    grads.emplace_back(std::vector<T>((*it).size()));
                }
                std::vector< std::vector<T> > changes;
                for (auto & w: weight_layers) {
                    changes.emplace_back(std::vector<T>(w.size()));
                }

                for (auto it_p = patterns_first, it_t = targets_first;
                     it_p != patterns_last; it_p++, it_t++) {
                    // Compute gradients first
                    // Calculate the errors: difference between target and pattern
                    calc_output(it_p->begin(), it_p->end());
                    for (i = 0; i < it_t->size(); i++) {
                        T out = neuron_layers.back()[i];
                        grads.back()[i] = ((*it_t)[i] - out);
                    }

                    // Step through the layers from output to first hidden layer
                    for (l = neuron_layers.size() - 1; l > 0; l--) {
                        // curr is the neuron layer we are working with
                        auto & curr = neuron_layers[l];
                        // prev is the neuron layer feeding curr
                        auto & prev = neuron_layers[l - 1];
                        // weights are the connections between prev and curr
                        auto & weights = weight_layers[l - 1];

                        // We step through each neuron in the current layer
                        for (size_t i = 0; i < curr.size(); i++) {
                            // Derivative of the error
                            T deriv = curr[i] * (1.0 - curr[i]);
                            // The backpropagated sum of errors times the derivative
                            grads[l-1][i] *= deriv;
                            // We step through the neurons of the previous layer
                            if (l > 1) {
                                for (j = 0; j < prev.size(); j++) {
                                    // This is the index into the weight connections
                                    // between the previous layer and the
                                    // current layer.
                                    // Currently this presumes fully connected.
                                    k = j * curr.size() + i;
                                    // Add the error
                                    grads[l-2][j] += weights[k] * grads[l-1][i];
                                }
                            }
                        }
                    }
                    // Now accumulate deltas (we can go forward this time)
                    for (l = 0; l < neuron_layers.size() - 1; l++) {
                        for (i = 0; i < neuron_layers[l].size(); i++) {
                            for (j = 0; j < neuron_layers[l + 1].size(); j++) {
                                k = i * neuron_layers[l+1].size() + j;
                                T change = eta * neuron_layers[l][i] * grads[l][j];
                                changes[l][k] += change;
                            }
                        }
                        // bias weights
                        k = neuron_layers[l].size() * neuron_layers[l+1].size();
                        for (i = 0; i < neuron_layers[l+1].size(); i++) {
                            T change = -eta * grads[l][i];
                            changes[l][k] += change;
                            k++;
                        }
                    }
                }
                for (i = 0; i < weight_layers.size(); i++) {
                    for (j = 0; j < weight_layers[i].size(); j++) {
                        weight_layers[i][j] += changes[i][j];
                    }
                }
            }
        }

        /**
         * Uses so-called batch training method on a set of patterns and
         * targets, in which weights are updated after all the patterns are
         * presented. The method works on fully connected nets using the sigmoid
         * function.
         *
         * @param patterns Set of patterns to train with
         *
         * @param targets Set of targets for corresponding patterns that net must be
         * trained to match. Number of targets must equal number of patterns.
         *
         * @param eta Factor by which to multiply the weight update deltas
         *
         * @param iterations Number of iterations to train for
         *
         */
        template <typename Container>
        void train_batch(const Container & patterns,
                         const Container & targets,
                         float eta,
                         unsigned int iterations) {
            train_batch(patterns.begin(), patterns.end(), targets.begin(),
                        eta, iterations);
        }

        /**
         * Function that dispatches the appropriate training method.
         *
         * @param patterns_first Iterator pointing to first pattern in set
         *
         * @param patterns_last Iterator pointing beyond the last pattern in set
         *
         * @param targets_first Iterator pointing to the first target (i.e. the
         * first output vector that the net must be trained to match on the
         * corresponding pattern) in a set. There must be an equal number of
         * patterns and targets.
         *
         * @param eta Factor by which to multiply the weight update deltas
         *
         * @param momentum Factor by which to multiply the previous weight
         *                 change on the new update (only relevant to online
         *                 training)
         *
         * @param iterations Number of iterations to train for
         *
         * @param online Whether train online (true) or batch (false)
         *
         */
        template <typename Iterator>
        void train(Iterator patterns_first,
                   Iterator patterns_last,
                   Iterator targets_first,
                   float eta,
                   float momentum,
                   unsigned int iterations,
                   bool online=true) {
            if (online) {
                train_online(patterns_first, patterns_last, targets_first,
                             eta, momentum, iterations);
            } else {
                train_batch(patterns_first, patterns_last, targets_first,
                            eta, iterations);
            }
        }

        /*
         * Function that dispatches the appropriate training method.
         *
         * @param patterns Set of patterns to train with
         *
         * @param targets Set of targets for corresponding patterns that net must be
         * trained to match. Number of targets must equal number of patterns.
         *
         * @param eta Factor by which to multiply the weight update deltas
         *
         * @param momentum Factor by which to multiply the previous weight
         *                 change on the new update (only relevant to online
         *                 training)
         *
         * @param iterations Number of iterations to train for
         *
         * @param online Whether train online (true) or batch (false)
         *
         */
        template <typename Container>
        void train(const Container & patterns,
                   const Container & targets,
                   float eta,
                   float momentum,
                   unsigned int iterations,
                   bool online=true) {
            train(patterns.begin(), patterns.end(), targets.begin(),
                  eta, momentum, iterations, online);
        }

        /*
         * Calculates the mean squared error for a net for given set of patterns and
         * their expected outputs (targets)
         *
         * @param patterns_first Iterator pointing to first pattern in set
         *
         * @param patterns_last Iterator pointing beyond the last pattern in set
         *
         * @param targets_first Iterator pointing to the first target in the set
         *
         */
        template <typename Iterator>
        double mean_squared_error(Iterator patterns_first,
                                  Iterator patterns_last,
                                  Iterator targets_first) {
            double total = 0.0, err;
            auto n = patterns_last - patterns_first;

            for (auto it_p = patterns_first, it_t = targets_first;
                 it_p != patterns_last; it_p++, it_t++) {
                calc_output(it_p->begin(), it_p->end());
                auto &out = neuron_layers.back();
                for (size_t j = 0; j < out.size(); j++) {
                    err = (*it_t)[j] - out[j];
                    total += err * err;
                }
            }
            return total / n;
        }

        /**
         * Calculates the mean squared error (MSE) for a net for given set of
         * patterns and their expected outputs (targets)
         *
         * @param patterns Set of patterns to calculate MSE for
         *
         * @param targets Set of targets for corresponding patterns
         *
         */
        template <typename Container>
        double mean_squared_error(const Container & patterns,
                                  const Container & targets) {
            return mean_squared_error(patterns.begin(), patterns.end(),
                                      targets.begin());
        }

        /**
         * Calculates the classification score for a net for given set of
         * patterns and their expected outputs (targets)
         *
         * @param patterns_first Iterator pointing to first pattern in set
         *
         * @param patterns_last Iterator pointing beyond the last pattern in set
         *
         * @param targets_first Iterator pointing to the first target in set
         *
         */
        template <typename Iterator>
        unsigned int score(Iterator patterns_first,
                           Iterator patterns_last,
                           Iterator targets_first) {
            unsigned errors = 0;
            auto n = patterns_last - patterns_first;
            for (auto it_p = patterns_first, it_t = targets_first;
                 it_p != patterns_last; it_p++, it_t++) {
                calc_output(it_p->begin(), it_p->end());
                auto &out = neuron_layers.back();
                for (size_t j = 0; j < out.size(); j++) {
                    if (std::round(out[j]) != (*it_t)[j]) {
                        ++errors;
                        break;
                    }
                }
            }
            return n - errors;
        }

        /**
         * Calculates the number of correct classifications for a net for given
         * set of patterns and their expected outputs (targets)
         *
         * @param patterns Set of patterns to calculate score for
         *
         * @param targets Set of targets for corresponding patterns
         *
         */
        template <typename Container>
        unsigned int score(const Container & patterns,
                           const Container & targets) {
            return score(patterns.begin(), patterns.end(), targets.begin());
        }

        void print(bool print_weights=true, bool print_neurons=false);
        void save(const char *filename);
        /// Returns the number of output neurons
        inline size_t num_outputs() const {
            return neuron_layers.back().size();
        }
        inline T output(size_t neuron_index) {
            return neuron_layers.back()[neuron_index];
        }
        /// Returns the random number generator
        inline std::mt19937 & get_rng() {return rng_;}

    protected:
        std::mt19937 rng_;

        /**
         * Constructor that simply calls the init function.
         *
         * @param parameters Specification for each layer of the net
         *
         * @param seed Used to seed the random generator (0 for random device to
         * be used)
         */
        void
        init(const std::vector< LayerParameters<T> > & parameters, int seed = 0) {
            std::random_device rd;
            assert(parameters.size() > 1);
            if (seed == 0) seed = rd();
            rng_.seed(seed);
            // Create layers of neurons
            for (auto p: parameters) {
                assert(p.neurons > 0);
                std::vector<T> neurons(p.neurons, (T) 0.0);
                neuron_layers.emplace_back(neurons);
            }
            // Create weights
            for (size_t i = 1; i < neuron_layers.size(); i++) {
                size_t num_weights = parameters[i].weights;
                if (num_weights == 0) {
                    // Fully connected plus bias neuron weights
                    num_weights = neuron_layers[i].size() *
                        neuron_layers[i-1].size() + neuron_layers[i].size();
                }
                std::vector<T> weights(num_weights);
                double min = (double) parameters[i].min_weight;
                double max = (double) parameters[i].max_weight;
                std::uniform_real_distribution<double> dist(min, max);
                for (auto & w: weights) {
                    w = (T) dist(rng_);
                }
                weight_layers.emplace_back(weights);
                std::vector<T> w(weights.size());
                weight_changes.emplace_back(w);
            }
        }

        /**
         * Feeds the output of one layer to the next layer.
         *
         * @param from The layer from which the output is coming
         * @param to The layer which receives the output
         * @param weights The weights connecting the two layers
         */
        void feed_forward(const std::vector<T> & from,
                          std::vector<T> & to,
                          const std::vector<T> & weights) {
            size_t i, j, k;
            T total;
            for (i = 0; i < to.size(); i++) {
                total = 0.0;
                for (j = 0; j < from.size(); j++) {
                    k = j * to.size() + i;
                    total += from[j] * weights[k];
                }
                // bias weight
                k = weights.size() - to.size() + i;
                to[i] = total - weights[k];
            }
        }
    };

    template <typename T> void
    print_vector(const std::vector<T> & vec,
                 const std::string & delim = "\n");

    template <typename T> void
    print_matrix(const std::vector< std::vector<T> > & matrix);

    template <typename T>
    Net<T> load(const char *filename);
}

#endif
