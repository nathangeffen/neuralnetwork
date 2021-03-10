/**
 * @file nn.cc
 * Defines the core neural network functions that are declared in nn.hh in the
 * nn namespace. Most of the definitions are for methods in the Net class.
 */

#include "nn_net.hh"



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
