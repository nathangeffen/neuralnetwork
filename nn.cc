/**
 * @file nn.cc
 * Defines the core neural network functions that are declared in nn.hh in the
 * nn namespace. Most of the definitions are for methods in the Net class.
 */

#include "nn.hh"
#include "dp.hh"

/**
 * The iconic multilayered pereceptron firing function
 * @param f The sum of the inputs to a neuron
 */

template <typename T>
T nn::sigmoid(T f)
{
    return (T) 1.0 / (1.0 + exp(-f));
}

/**
 * Executes sigmoid on a bunch of neurons
 * @param vec Usually a layer of neurons that have been summed
 */

template <typename T>
void nn::sigmoid_vector(std::vector<T> & vec)
{
    size_t i;
    const auto n = vec.size();
    T *data = vec.data();
#pragma omp parallel for shared(data) private(i)
    for (i = 0; i < n; i++) {
        data[i] = sigmoid<T>(data[i]);
    }
}

/**
 * There should be a set of parameters for each layer.
 * The input layer parameters are only used to determine number of neurons.
 * i.e. The number of weights etc in the input layer are ignored.
 *
 * @param parameters Specification for each layer of the net
 * @param seed Used to seed the random generator (0 for random device to be used)
 */

template <typename T>
void nn::Net<T>::init(const std::vector< LayerParameters<T> > & parameters,
                      int seed)
{
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
 * Constructor that simply calls the init function.
 *
 * @param parameters Specification for each layer of the net
 * @param seed Used to seed the random generator (0 for random device to be used)
 */

template <typename T>
nn::Net<T>::Net(const std::vector< nn::LayerParameters<T> > & parameters, int seed)
{
    init(parameters, seed);
}


/**
 * Default constructor. Does nothing.
 */

template <typename T>
nn::Net<T>::Net()
{
};

/**
 * Feeds the output of one layer to the next layer.
 *
 * @param from The layer from which the output is coming
 * @param to The layer which receives the output
 * @param weights The weights connecting the two layers
 */

template <typename T>
void
nn::Net<T>::feed_forward(const std::vector<T> & from,
                         std::vector<T> & to,
                         const std::vector<T> & weights)
{
    size_t i, j, k;
    T total;
    for (i = 0; i < to.size(); i++) {
        total = 0.0;
// #pragma omp parallel for private(j, k) shared(from, weights) reduction(+:total)
        for (j = 0; j < from.size(); j++) {
            k = j * to.size() + i;
            total += from[j] * weights[k];
        }
        // bias weight
        k = weights.size() - to.size() + i;
        to[i] = total - weights[k];
    }
}

/**
 * Calculates the net's output for a given input pattern
 *
 * @param first The first entry of the input pattern
 * @param last The last entry of the input pattern
 *
 * @return The output layer of the neural net
 */

template <typename T>
std::vector<T>
nn::Net<T>::output(typename std::vector<T>::const_iterator first,
                   typename std::vector<T>::const_iterator last)
{
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

template <typename T>
std::vector<T> output(const std::vector<T> & pattern) {
    output(pattern.begin(), pattern.end());
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

template <typename T>
void nn::Net<T>::train_single(typename std::vector<T>::const_iterator pattern_first,
                              typename std::vector<T>::const_iterator pattern_last,
                              typename std::vector<T>::const_iterator target_first,
                              typename std::vector<T>::const_iterator target_last,
                              float eta,
                              float momentum)
{
    size_t i, j, k, l;
    T change;
    assert(neuron_layers.size() > 2);
    assert( (size_t) (pattern_last - pattern_first) ==
            neuron_layers.front().size());
    assert( (size_t) (target_last - target_first) == neuron_layers.back().size());

    // Calculate the errors: difference between target and pattern
    output(pattern_first, pattern_last);
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
// #pragma omp parallel for shared(prev_deltas, weights, momentum, delta, eta, n)
//    private(j, k, change)
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
 * Executes the backpropagation algorithm on a single training-target pair. The
 * result is stored in the last (output) layer of the net.
 *
 * @param pattern Pattern on which to train the net
 * @param target Expected output for the pattern
 * @param eta Factor by which to multiply the weight update deltas
 * @param momentum Factor by which to multiply the previous weight change
 *                 on the new update
 *
 */

template <typename T>
void nn::Net<T>::train_single(const std::vector<T> & pattern,
                              const std::vector<T> & target,
                              float eta,
                              float momentum)
{
    train_single(pattern.begin(), pattern.end(), target.begin(), target.end(),
                 eta, momentum);
}

/**
 * Uses so-called online training method on a set of patterns and targets,
 * in which weights are updated after each pattern is presented.
 *
 * @param patterns_first Iterator pointing to first pattern in set
 *
 * @param patterns_last Iterator pointing beyond the last pattern in set
 *
 * @param targets_first Iterator pointing to the first target (i.e. the first
 * output vector that the net must be trained to match on the corresponding
 * pattern) in a set. There must be an equal number of patterns and targets.
 *
 * @param eta Factor by which to multiply the weight update deltas
 *
 * @param momentum Factor by which to multiply the previous weight change
 *                 on the new update
 *
 * @param iterations Number of iterations to train for
 *
 */


template <typename T>
void nn::Net<T>::train_online(
    typename std::vector< std::vector<T> >::const_iterator patterns_first,
    typename std::vector< std::vector<T> >::const_iterator patterns_last,
    typename std::vector< std::vector<T> >::const_iterator targets_first,
    float eta, float momentum,
    unsigned int iterations)
{
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
 * Uses so-called online training method on a set of patterns and targets,
 * in which weights are updated after each pattern is presented.
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

template <typename T>
void nn::Net<T>::train_online(const std::vector< std::vector<T> > & patterns,
                              const std::vector< std::vector<T> > & targets,
                              float eta, float momentum, unsigned int iterations)
{
    train_online(patterns.begin(), patterns.end(), targets.begin(),
                 eta, momentum, iterations);
}

/**
 * Uses so-called batch training method on a set of patterns and targets, in
 * which weights are updated after all the patterns are presented. The method works
 * on fully connected nets using the sigmoid function.
 *
 * @param patterns_first Iterator pointing to first pattern in set
 *
 * @param patterns_last Iterator pointing beyond the last pattern in set
 *
 * @param targets_first Iterator pointing to the first target (i.e. the first
 * output vector that the net must be trained to match on the corresponding
 * pattern) in a set. There must be an equal number of patterns and targets.
 *
 * @param eta Factor by which to multiply the weight update deltas
 *
 * @param iterations Number of iterations to train for
 *
 */

template <typename T>
void nn::Net<T>::train_batch(
            typename std::vector< std::vector<T> >::const_iterator patterns_first,
            typename std::vector< std::vector<T> >::const_iterator patterns_last,
            typename std::vector< std::vector<T> >::const_iterator targets_first,
            float eta,
            unsigned int iterations)
{
    size_t i, j, k, l;
    unsigned int c;
    assert(neuron_layers.size() > 2);
    assert(patterns_last - patterns_first > 0);
    assert(patterns_first->size() == neuron_layers.front().size());
    assert(neuron_layers.back().size() == targets_first->size());

    for (c = 0; c < iterations; c++) {
        std::vector< std::vector<T> > grads;
        for (auto it = neuron_layers.begin() + 1; it < neuron_layers.end(); it++) {
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
            output(it_p->begin(), it_p->end());
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
                            // between the previous layer and the current layer.
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
 * Uses so-called batch training method on a set of patterns and targets, in
 * which weights are updated after all the patterns are presented. The method works
 * on fully connected nets using the sigmoid function.
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

template <typename T>
void nn::Net<T>::train_batch(const std::vector< std::vector<T> > & patterns,
                             const std::vector< std::vector<T> > & targets,
                             float eta,
                             unsigned int iterations)
{
    train_batch(patterns.begin(), patterns.end(), targets.begin(),
                eta, iterations);
}

/*
 * Function that dispatches the appropriate training method.
 *
 * @param patterns_first Iterator pointing to first pattern in set
 *
 * @param patterns_last Iterator pointing beyond the last pattern in set
 *
 * @param targets_first Iterator pointing to the first target (i.e. the first
 * output vector that the net must be trained to match on the corresponding
 * pattern) in a set. There must be an equal number of patterns and targets.
 *
 * @param eta Factor by which to multiply the weight update deltas
 *
 * @param momentum Factor by which to multiply the previous weight change on the
 *                 new update (only relevant to online training)
 *
 * @param iterations Number of iterations to train for
 *
 * @param online Whether train online (true) or batch (false)
 *
 */

template <typename T>
void nn::Net<T>::train(
    typename std::vector< std::vector<T> >::const_iterator patterns_first,
    typename std::vector< std::vector<T> >::const_iterator patterns_last,
    typename std::vector< std::vector<T> >::const_iterator targets_first,
    float eta,
    float momentum,
    unsigned int iterations,
    bool online)
{
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
 * @param momentum Factor by which to multiply the previous weight change on the
 *                 new update (only relevant to online training)
 *
 * @param iterations Number of iterations to train for
 *
 * @param online Whether train online (true) or batch (false)
 *
 */

template <typename T>
void nn::Net<T>::train(const std::vector< std::vector<T> > & patterns,
                       const std::vector< std::vector<T> > & targets,
                       float eta,
                       float momentum,
                       unsigned int iterations,
                       bool online)
{
    assert(patterns.size() == targets.size());
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

template <typename T>
double nn::Net<T>::mean_squared_error(
            typename std::vector< std::vector<T> >::const_iterator patterns_first,
            typename std::vector< std::vector<T> >::const_iterator patterns_last,
            typename std::vector< std::vector<T> >::const_iterator targets_first)
{
    double total = 0.0, err;
    auto n = patterns_last - patterns_first;

    for (auto it_p = patterns_first, it_t = targets_first;
         it_p != patterns_last; it_p++, it_t++) {
        output(it_p->begin(), it_p->end());
        auto &out = neuron_layers.back();
        for (size_t j = 0; j < out.size(); j++) {
            err = (*it_t)[j] - out[j];
            total += err * err;
        }
    }

    return total / n;
}
/*
 * Calculates the mean squared error (MSE) for a net for given set of patterns and
 * their expected outputs (targets)
 *
 * @param patterns Set of patterns to calculate MSE for
 *
 * @param targets Set of targets for corresponding patterns
 *
 */

template <typename T>
double nn::Net<T>::mean_squared_error(
    const std::vector< std::vector<T> > & patterns,
    const std::vector< std::vector<T> > & targets)
{
    return mean_squared_error(patterns.begin(), patterns.end(), targets.begin());
}

/*
 * Calculates the classification score for a net for given set of patterns and
 * their expected outputs (targets)
 *
 * @param patterns_first Iterator pointing to first pattern in set
 *
 * @param patterns_last Iterator pointing beyond the last pattern in set
 *
 * @param targets_first Iterator pointing to the first target in set
 *
 */

template <typename T>
unsigned int nn::Net<T>::score(
    typename std::vector< std::vector<T> >::const_iterator patterns_first,
    typename std::vector< std::vector<T> >::const_iterator patterns_last,
    typename std::vector< std::vector<T> >::const_iterator targets_first)
{
    unsigned errors = 0;
    auto n = patterns_last - patterns_first;
    for (auto it_p = patterns_first, it_t = targets_first;
         it_p != patterns_last; it_p++, it_t++) {
        output(it_p->begin(), it_p->end());
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

/*
 * Calculates the number of correct classifications for a net for given set of
 * patterns and their expected outputs (targets)
 *
 * @param patterns Set of patterns to calculate score for
 *
 * @param targets Set of targets for corresponding patterns
 *
 */

template <typename T>
unsigned int nn::Net<T>::score(
    const std::vector< std::vector<T> > & patterns,
    const std::vector< std::vector<T> > & targets)
{
    return score(patterns.begin(), patterns.end(), targets.begin());
}

/*
 * Loads a neural net from a file.
 *
 * @param filename The name of the file to read from
 */

template <typename T>
nn::Net<T> nn::load(const char *filename)
{
    nn::Net<T> net;
    std::vector< LayerParameters<T> > parameters;
    std::string line;
    std::vector<size_t> neuron_sizes;
    std::vector<size_t> weight_sizes;
    std::vector<T> weights;
    size_t val;

    std::ifstream input(filename, std::ifstream::in);
    if (!input) {
        throw std::runtime_error(std::string("Can't open input file.")
                                 + filename);
    }

    if (std::getline(input, line)) {
        std::istringstream in(line);
        while(in >> val) neuron_sizes.push_back(val);
        if (neuron_sizes.size() == 0) {
            throw std::runtime_error("No neurons line specified in net file.");
        }
    } else {
        throw std::runtime_error("No neurons line in net file.");
    }

    if (std::getline(input, line)) {
        std::istringstream in(line);
        while(in >> val) weight_sizes.push_back(val);
    } else {
        throw std::runtime_error("No weights line in net file.");
    }

    if (weight_sizes.size() != neuron_sizes.size()) {
        throw std::runtime_error("Mismatch between weights and neurons");
    }

    for (size_t i = 0; i < neuron_sizes.size(); i++) {
        LayerParameters<T> p;
        p.neurons = neuron_sizes[i];
        p.weights = weight_sizes[i];
        parameters.push_back(p);
    }
    net.init(parameters);

    for (auto &l: net.weight_layers) {
        for (auto &w: l) {
            input >> w;
            if (input.eof()) {
                throw std::runtime_error("Too few weights in net file.");
            }
        }
    }
    return net;
}

/*
 * Saves a neural net to a file.
 *
 * @param filename Name of file to save the net to
 */

template <typename T>
void
nn::Net<T>::save(const char *filename)
{
    std::ofstream output(filename);
    for (auto n: neuron_layers) output << n.size() << ' ';
    output << '\n';
    output << 0  << ' ';
    for (auto w: weight_layers) output << w.size() << ' ';
    output << '\n';
    for (auto &l: weight_layers) {
        for (auto &w: l) {
            output << w << '\n';
        }
    }
}

/*
 * Prints the elements of a vector (as a row)
 *
 * @param vec The vector to print
 * @param delim The separator between each element
 */

template <typename T>
void nn::print_vector(const std::vector<T> & vec, const std::string & delim)
{
    for (auto f: vec)
        std::cout << f << delim;
}

/*
 * Prints the elements of a matrix
 *
 * @param matrix The matrix to print
 */


template <typename T>
void nn::print_matrix(const std::vector< std::vector<T> > & matrix)
{
    for (auto v: matrix) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

/*
 * Displays details about a neural net
 *
 * @param print_weights Whether to print weight information (true) or not (false)
 * @param print_neurons Whether to print neuron outputs (true) or not (false)
 *
 */

template <typename T>
void nn::Net<T>::print(bool print_weights, bool print_neurons)
{
    size_t i, j, k, l;
    for (i = 1; i < neuron_layers.size(); i++) {
        std::cout << "Layer ";
        if (i - 1 == 0) {
            std::cout << "input";
        } else {
            std::cout << "hidden " << i - 1;
        }
        std::cout << " to ";
        if (i == neuron_layers.size() - 1) {
            std::cout << "output" << std::endl;
        } else {
            std::cout << "hidden " << i << std::endl;
        }
        for (j = 0; j < neuron_layers[i-1].size(); j++) {
            if (print_neurons) {
                std::cout << "Neuron " << i-1 << '-' << j << ": "
                          << neuron_layers[i-1][j] << std::endl;
            }
            if (print_weights) {
                for (k = 0; k < neuron_layers[i].size(); k++) {
                    l = j * neuron_layers[i].size() + k;
                    std::cout << i-1 << '-' << j << "\tto\t"
                              << i << '-' << k << " (" << l << "): "
                              << weight_layers[i-1][l] << std::endl;
                }
            }
        }
        // Bias weights
        if (print_weights) {
            for (k = 0; k < neuron_layers[i].size(); k++) {
                l = weight_layers[i-1].size() - neuron_layers[i].size() + k;
                std::cout << "BIAS\tto\t" <<  i << '-' << k
                          << " (" << l << "): "
                          << weight_layers[i-1][l] << std::endl;
            }
        }
    }
    // Print output neurons
    if (print_neurons) {
        std::cout << "Output" << std::endl;
        k = neuron_layers.size() - 1;
        for (j = 0; j < neuron_layers[k].size(); j++) {
            std::cout << "Neuron " << k << '-' << j << ": "
                      << neuron_layers[k][j] << std::endl;
        }
    }
}

template class nn::Net<float>;
template class nn::Net<double>;
template class nn::Net<int8_t>;
