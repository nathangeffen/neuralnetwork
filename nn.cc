#include "nn.hh"
#include "dp.hh"

template <typename T>
T nn::sigmoid(T f)
{
    return (T) 1.0 / (1.0 + exp(-f));
}

template <typename T>
void nn::sigmoid_vector(std::vector<T> & vec)
{
#pragma omp parallel for
    for (auto it = vec.begin(); it != vec.end(); it++)
        *it = sigmoid<T>(*it);
}

/*
 * There should be a set of parameters for each layer.
 * The input layer parameters are only used to determine number of neurons.
 * i.e. The number of weights etc in the input layer are ignored.
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
        std::uniform_real_distribution<T> dist(parameters[i].min_weight,
                                               parameters[i].max_weight);
        for (auto & w: weights) {
            w = dist(rng_);
        }
        weight_layers.emplace_back(weights);
        std::vector<T> w(weights.size());
        weight_changes.emplace_back(w);
    }
}

template <typename T>
nn::Net<T>::Net(const std::vector< nn::LayerParameters<T> > & parameters, int seed)
{
    init(parameters, seed);
}

template <typename T>
nn::Net<T>::Net()
{
};

template <typename T>
void
nn::Net<T>::feed_forward(std::vector<T> & from,
                         std::vector<T> & to,
                         std::vector<T> & weights)
{
    size_t i, j, k;
    T total;
//#pragma omp parallel for private(total, i)
    for (i = 0; i < to.size(); i++) {
        total = 0.0;
#pragma omp parallel for private(j, k) reduction(+:total)
        for (j = 0; j < from.size(); j++) {
            k = j * to.size() + i;
            total += from[j] * weights[k];
        }
        // bias weight
        k = weights.size() - to.size() + i;
        to[i] = total - weights[k];
    }
}

template <typename T>
std::vector<T>
nn::Net<T>::output(const std::vector<T> & pattern)
{
    assert(pattern.size() == neuron_layers[0].size());
    std::copy(pattern.begin(), pattern.end(), neuron_layers[0].begin());
    for (size_t i = 1; i < neuron_layers.size(); i++) {
        feed_forward(neuron_layers[i-1], neuron_layers[i],
                     weight_layers[i-1]);
        sigmoid_vector(neuron_layers[i]);
    }
    return neuron_layers.back();
}

template <typename T>
void nn::Net<T>::train_single(const std::vector<T> & pattern,
                              const std::vector<T> & target,
                              float eta,
                              float momentum)
{
    size_t i, j, k, l;
    T change;
    assert(neuron_layers.size() > 2);
    assert(pattern.size() == neuron_layers.front().size());
    assert(neuron_layers.back().size() == target.size());

    // Calculate the errors: difference between target and pattern
    output(pattern);
    std::vector<T> curr_deltas(neuron_layers.back().size());
    for (i = 0; i < target.size(); i++) {
        curr_deltas[i] = target[i] - neuron_layers.back()[i];
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
        for (size_t i = 0; i < curr.size(); i++) {
            // Derivative of the error
            T deriv = curr[i] * (1.0 - curr[i]);
            // The backpropagated sum of errors times the derivative
            T delta = curr_deltas[i] * deriv;
            // We step through the neurons of the previous layer
#pragma omp parallel for shared(prev_deltas, weights) private(j, k, change)
            for (j = 0; j < prev.size(); j++) {
                // This is the index into the weight connections
                // between the previous layer and the current layer.
                // Currently this presumes fully connected.
                k = j * curr.size() + i;
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

template <typename T>
void nn::Net<T>::train_online(const std::vector< std::vector<T> > & patterns,
                              const std::vector< std::vector<T> > & targets,
                              float eta, float momentum,
                              unsigned int iterations)
{
    assert(patterns.size() == targets.size());
    std::vector<size_t>  indices(patterns.size());
    for (size_t i = 0; i < indices.size(); i++) indices[i] = i;
    for (unsigned int i = 0; i < iterations; i++) {
        std::shuffle(indices.begin(), indices.end(), rng_);
        for (auto index: indices) {
            train_single(patterns[index], targets[index], eta, momentum);
        }
    }
}

template <typename T>
void nn::Net<T>::train_batch(const std::vector< std::vector<T> > & patterns,
                             const std::vector< std::vector<T> > & targets,
                             float eta,
                             float momentum,
                             unsigned int iterations)
{
    size_t i, j, k, l, p;
    unsigned int c;
    assert(neuron_layers.size() > 2);
    assert(patterns.size() > 0);
    assert(patterns.size() == targets.size());
    assert(patterns[0].size() == neuron_layers.front().size());
    assert(neuron_layers.back().size() == targets[0].size());

    for (c = 0; c < iterations; c++) {
        std::vector< std::vector<T> > grads;
        for (auto it = neuron_layers.begin() + 1; it < neuron_layers.end(); it++) {
            grads.emplace_back(std::vector<T>((*it).size()));
        }
        std::vector< std::vector<T> > changes;
        for (auto & w: weight_layers) {
            changes.emplace_back(std::vector<T>(w.size()));
        }


        for (p = 0; p < patterns.size(); p++) {
            // Compute gradients first
            // Calculate the errors: difference between target and pattern
            output(patterns[p]);
            for (i = 0; i < targets[p].size(); i++) {
                T out = neuron_layers.back()[i];
                grads.back()[i] = (targets[p][i] - out);
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

template <typename T>
void nn::Net<T>::train(const std::vector< std::vector<T> > & patterns,
                       const std::vector< std::vector<T> > & targets,
                       float eta,
                       float momentum,
                       unsigned int iterations,
                       bool online)
{
    if (online) {
        train_online(patterns, targets, eta, momentum, iterations);
    } else {
        train_batch(patterns, targets, eta, momentum, iterations);
    }

}

template <typename T>
double nn::Net<T>::mean_squared_error(
    const std::vector< std::vector<T> > & patterns,
    const std::vector< std::vector<T> > & targets)
{
    assert(patterns.size() == targets.size());
    double total = 0.0, err;
    for (size_t i = 0; i < patterns.size(); i++) {
        output(patterns[i]);
        auto &out = neuron_layers.back();
        for (size_t j = 0; j < out.size(); j++) {
            err = targets[i][j] - out[j];
            total += err * err;
        }
    }
    return total / patterns.size();
}

template <typename T>
unsigned int nn::Net<T>::score(
    const std::vector< std::vector<T> > & patterns,
    const std::vector< std::vector<T> > & targets)
{
    assert(patterns.size() == targets.size());
    unsigned errors = 0;
    for (size_t i = 0; i < patterns.size(); i++) {
        output(patterns[i]);
        auto &out = neuron_layers.back();
        for (size_t j = 0; j < out.size(); j++) {
            if (std::round(out[j]) != targets[i][j]) {
                ++errors;
                break;
            }
        }
    }
    return patterns.size() - errors;
}

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

template <typename T>
void nn::print_vector(const std::vector<T> & vec, const std::string & delim)
{
    for (auto f: vec)
        std::cout << f << delim;
}

template <typename T>
void nn::print_matrix(const std::vector< std::vector<T> > & matrix)
{
    for (auto v: matrix) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

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



int main()
{
    std::vector< nn::LayerParameters<float> > parameters = {
        {
            .neurons = 2
        },
        {
            .neurons = 3
        },
        {
            .neurons = 4
        },
        {
            .neurons = 3
        },
        {
            .neurons = 2
        }
    };

    std::vector< std::vector<float> > patterns = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    };
    std::vector< std::vector<float> > targets = {
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };

    float eta = 0.2, momentum = 1.0;
    size_t iterations = 3000;

    nn::Net< float > mlp(parameters, 3);
    mlp.save("failed_xor.nn");
    for (auto &p: patterns) {
        auto v = mlp.output(p);
        for (auto n: v) std::cout << n << ' ';
        std::cout << '\n';
    }
    mlp.train(patterns, targets, eta, momentum, iterations);

    std::cout << "Output \n";

    for (auto &p: patterns) {
        auto v = mlp.output(p);
        for (auto n: v) std::cout << n << ' ';
        std::cout << '\n';
    }

    patterns = targets = {};
    for (float f = 0; f < 3.14; f += 0.01) {
        patterns.push_back({f});
        targets.push_back({ (float) sin(f)});
    }

    parameters = {
        {
            .neurons = 1
        },
        {
            .neurons = 2
        },
        {
            .neurons = 2
        },
        {
            .neurons = 1
        },

    };
    nn::Net<float> ann(parameters, 5);
    std::cout << "MSE sin: " << ann.mean_squared_error(patterns, targets) << '\n';
    ann.train(patterns, targets, eta, momentum, iterations);
    std::cout << "MSE sin: " << ann.mean_squared_error(patterns, targets) << '\n';

    std::ifstream input("iris.data", std::ifstream::in);
    dp::CSV<std::string> csv = dp::read_csv(input);
    dp::convert_csv_col_labels_to_ints(csv, csv.rows[0].size() - 1);
    dp::CSV<float> csv_f;
    dp::convert_csv_types(csv, csv_f);

    patterns = {};
    targets = {};
    std::cout << "Converting patterns\n";
    dp::convert_csv_patterns_targets(csv_f, patterns, targets, 3);
    std::cout << patterns.size() << ' ' << patterns[10].size() << ' '
              << targets.size() << ' ' << targets[10].size() << '\n';

    parameters = {
        {
            .neurons = 4
        },
        {
            .neurons = 4
        },
        {
            .neurons = 3
        }
    };
    nn::Net<float> iris(parameters, 7);
    std::cout << "Iris score: " << iris.score(patterns, targets) << '\n';
    iris.train(patterns, targets, eta, momentum, iterations);
    std::cout << "Iris score: " << iris.score(patterns, targets) << '\n';


    // std::cout << "Stress\n";
    // nn::Net<float> ann(parameters, 5);
    // std::vector<float> pattern;
    // std::mt19937 rng;
    // std::uniform_real_distribution<float> dist(0.0, 1.0);
    // for (size_t i = 0; i < n_input; i++) {
    //     pattern.push_back(dist(rng));
    // }
    // ann.output(pattern);
    // for (auto o: ann.neuron_layers.back()) std::cout << o << '\t';
    // std::cout << '\n';

    return 0;
}
