#include <iostream>
#include <string>
#include <cstdlib>
#include "nn.hh"


template<typename T=float, typename IteratorType>
std::string sliceToString(IteratorType first, IteratorType last,
                          const std::string& delim=", ",
                          const std::string& prefix="",
                          const std::string& suffix="")
{
    std::stringstream ss;
    ss << prefix;
    auto it = first;
    for (; it != last - 1; it++) {
        ss << *it << delim;
    }
    ss << *it;
    ss << suffix;
    return ss.str();
}


int main()
{

    // Read data and covert to matrix
    std::ifstream input("abalone.csv", std::ifstream::in);
    nn::CSV<std::string> csv = nn::read_csv(input);
    nn::convert_csv_col_labels_to_ints(csv, 0);
    nn::Matrix<float> matrix = nn::csv_to_matrix<float>(csv);
    std::mt19937 rng;
    std::shuffle(matrix.begin(), matrix.end(), rng);
    size_t n_training = (double) 0.5 * matrix.num_rows();

    // Convert to training and testing patterns and targets
    auto pattern_train_matrix = sub_matrix(matrix,
                                           0, 0,
                                           n_training, matrix.num_cols() - 1);
    auto target_train_matrix = sub_matrix(matrix,
                                          0, matrix.num_cols() - 1,
                                          n_training, matrix.num_cols());
    auto pattern_test_matrix = sub_matrix(matrix,
                                          n_training, 0,
                                          matrix.num_rows(), matrix.num_cols() - 1);
    auto target_test_matrix = sub_matrix(matrix,
                                          n_training, matrix.num_cols() - 1,
                                          matrix.num_rows(), matrix.num_cols());
    // Normalize the data (min and max only calculated on training data)
    auto pattern_min_max = pattern_train_matrix.min_max_cols();
    auto target_min_max = target_train_matrix.min_max_cols();

    auto pattern_train_norm_matrix = pattern_train_matrix.normalize_cols(
        0.0, 1.0, pattern_min_max);
    auto target_train_norm_matrix = target_train_matrix.normalize_cols(
        0.0, 1.0, target_min_max);
    auto pattern_test_norm_matrix = pattern_test_matrix.normalize_cols(
        0.0, 1.0, pattern_min_max);
    auto target_test_norm_matrix = target_test_matrix.normalize_cols(
        0.0, 1.0, target_min_max);

    std::vector< nn::LayerParameters<float> > parameters = {
        {
            .neurons = 8
        },
        {
            .neurons = 8
        },
        {
            .neurons = 8
        },
        {
            .neurons = 1
        },
    };

    float eta = 0.2, momentum = 1.0;
    unsigned iterations = 3000;

    nn::Net<float> mlp(parameters);
    std::cout << "MSE Before: " << mlp.mean_squared_error(
        pattern_train_norm_matrix, target_train_norm_matrix) << '\n';
    mlp.train(pattern_train_norm_matrix, target_train_norm_matrix, eta, momentum,
              iterations);
    std::cout << "MSE After: " << mlp.mean_squared_error(
        pattern_train_norm_matrix, target_train_norm_matrix) << '\n';
    std::cout << "MSE for test: " << mlp.mean_squared_error(
        pattern_test_norm_matrix, target_test_norm_matrix) << '\n';

    float lo = target_min_max[0].first;
    float hi = target_min_max[0].second;

    unsigned n = pattern_train_matrix.num_rows();
    for (unsigned i = 0; i < n; i++) {
        auto out = mlp.calc_output(pattern_test_norm_matrix[i])[0];
        std::cout << "Output: "   << out;
        std::cout << " Expected: " << target_test_norm_matrix[i][0];
        std::cout << " Denormal output: " << out * (hi - lo) + lo;
        std::cout << " Denormal expected: " << target_test_matrix[i][0] << '\n';
    }

    // std::cout << "MSE sin: " << ann.mean_squared_error(patterns, targets) << '\n';
    // ann.train(patterns, targets, eta, momentum, iterations);
    // std::cout << "MSE sin: " << ann.mean_squared_error(patterns, targets) << '\n';

    // mlp.save("failed_xor.nn");
    // for (auto &p: patterns) {
    //     auto v = mlp.calc_output(p.begin(), p.end());
    //     for (auto n: v) std::cout << n << ' ';
    //     std::cout << '\n';
    // }
    // mlp.train(patterns, targets, eta, momentum, iterations);


    //std::cout << matrix;


    // std::vector<float> t = {1, 2, 3, 4, 3, 2, 1};
    // std::cout << "Mean: " << nn::mean(t.begin(), t.end()) << '\n';
    // std::cout << "stdev: " << nn::stdev(t.begin(), t.end()) << '\n';
    // nn::normalize<float>(t.begin(), t.end(), -1.0, 1.0);
    // std::cout << "normalize:\n" << sliceToString(t.begin(), t.end()) << '\n';

    // nn::Matrix<float> m(
    //     {
    //         {1.0, 2.0, 3.0},
    //         {4.0, 5.0, 6.0},
    //         {7.0, 8.0, 9.0},
    //         {10.0, 11.0, 12.0}
    //     });

    // std::cout << "MATRIX\n";
    // std::cout << m;

    // std::cout << "MATRIX COLUMN 1\n";
    // auto c = m.column(1);
    // for (auto it = c.begin(); it != c.end(); it++) {
    //     std::cout << *it << " ";
    // }
    // std::cout << "\n";
    // std::cout << c.end() - c.begin() << '\n';

    // std::cout << "TRANSPOSE\n";
    // nn::Matrix<float> tr = m.transpose();
    // std::cout << tr;

    // std::cout << "ADDITION\n";
    // std::cout << m + m;

    // std::cout << "MULT TRANSPOSE\n";
    // std::cout << m * tr;

    // nn::Matrix<std::complex<double>> mc(3,3);

    // std::mt19937 rng;
    // std::uniform_real_distribution<double> dist(-10.0, 10.0);
    // std::cout << "Complex: ";
    // mc.for_each(
    //     [&](std::complex<double> &val) {
    //         val = std::complex<double>(dist(rng), dist(rng));
    //     });
    // std::cout << mc * mc.transpose();
    // std::cout << "****\n";
    // std::cout << mc + mc;
    // std::cout << "****\n";
    // std::cout << mc - mc;
    // std::cout << "****\n";

    // std::cout << mc;
    // std::cout << "SCALAR PRE\n";
    // std::cout << 5.0 * mc;

    // std::cout << "SCALAR POST\n";
    // std::cout << mc * 5.0;

    // std::cout << "ALL ONES\n";
    // std::cout << nn::Matrix<float>(5, 3, 1);

    // std::cout << "IDENTITY\n";
    // std::cout << nn::Ident<float>(5);

    // std::vector< nn::LayerParameters<float> > parameters = {
    //     {
    //         .neurons = 2
    //     },
    //     {
    //         .neurons = 3
    //     },
    //     {
    //         .neurons = 4
    //     },
    //     {
    //         .neurons = 3
    //     },
    //     {
    //         .neurons = 2
    //     }
    // };

    // std::vector< std::vector<float> > patterns = {
    //     {0.0, 0.0},
    //     {1.0, 0.0},
    //     {0.0, 1.0},
    //     {1.0, 1.0}
    // };
    // std::vector< std::vector<float> > targets = {
    //     {0.0, 1.0},
    //     {1.0, 0.0},
    //     {1.0, 0.0},
    //     {0.0, 1.0}
    // };

    // float eta = 0.2, momentum = 1.0;
    // size_t iterations = 3000;

    // nn::Net<float> mlp(parameters, 3);
    // mlp.save("failed_xor.nn");
    // for (auto &p: patterns) {
    //     auto v = mlp.calc_output(p.begin(), p.end());
    //     for (auto n: v) std::cout << n << ' ';
    //     std::cout << '\n';
    // }
    // mlp.train(patterns, targets, eta, momentum, iterations);

    // std::cout << "Output \n";

    // for (auto &p: patterns) {
    //     auto v = mlp.calc_output(p.begin(), p.end());
    //     for (auto n: v) std::cout << n << ' ';
    //     std::cout << '\n';
    // }

    // patterns = targets = {};
    // for (float f = 0; f < 3.14; f += 0.01) {
    //     patterns.push_back({f});
    //     targets.push_back({ (float) sin(f)});
    // }

    // parameters = {
    //     {
    //         .neurons = 1
    //     },
    //     {
    //         .neurons = 2
    //     },
    //     {
    //         .neurons = 2
    //     },
    //     {
    //         .neurons = 1
    //     },

    // };
    // nn::Net<float> ann(parameters, 5);
    // std::cout << "MSE sin: " << ann.mean_squared_error(patterns, targets) << '\n';
    // ann.train(patterns, targets, eta, momentum, iterations);
    // std::cout << "MSE sin: " << ann.mean_squared_error(patterns, targets) << '\n';

    // parameters = {
    //     {
    //         .neurons = 4
    //     },
    //     {
    //         .neurons = 5
    //     },
    //     {
    //         .neurons = 3
    //     }
    // };

    // iterations = 30000;
    // patterns = {};
    // targets = {};

    // nn::Net<float> iris(parameters, 7);
    // nn::csv_file_to_patterns_and_targets("iris.data", patterns, targets,
    //                                      true, {}, iris.get_rng());
    // std::cout << "Iris score: " << iris.score(patterns, targets) << '\n';
    // std::cout << "Iris mse: " << iris.mean_squared_error(patterns, targets) << '\n';

    // std::cout << "Patterns to Matrix\n";
    // auto patterns_matrix = nn::patterns_to_matrix<float>(patterns.begin(),
    //                                                      patterns.begin() + 50);

    // iris.train(patterns_matrix.begin(), patterns_matrix.begin() + 50 ,
    //            targets.begin(), eta, momentum, iterations);
    // std::cout << "Iris score train: " << iris.score(patterns.begin(),
    //                                                 patterns.begin() + 50,
    //                                                 targets.begin()) << '\n';
    // std::cout << "Iris mse train: "
    //           << iris.mean_squared_error(patterns.begin(),
    //                                      patterns.begin() + 50,
    //                                      targets.begin()) << '\n';


    // std::cout << "Iris score test: " << iris.score(patterns.begin() + 50,
    //                                                patterns.end(),
    //                                                targets.begin() + 50) << '\n';
    // std::cout << "Iris mse test: "
    //           << iris.mean_squared_error(patterns.begin() + 50,
    //                                      patterns.end(),
    //                                      targets.begin() + 50) << '\n';


    // // parameters = {
    // //     {
    // //         .neurons = 20000
    // //     },
    // //     {
    // //         .neurons = 20000
    // //     },
    // //     {
    // //         .neurons = 20000
    // //     },
    // //     {
    // //         .neurons = 5
    // //     }
    // // };

    // // std::cout << "Stress\n";
    // // nn::Net<float> stress(parameters, 7);
    // // std::vector<float> pattern;
    // // std::mt19937 rng;
    // // std::uniform_real_distribution<float> dist(0.0, 1.0);
    // // for (size_t i = 0; i < parameters[0].neurons; i++) {
    // //     pattern.push_back(dist(rng));
    // // }
    // // stress.calc_output(pattern.begin(), pattern.end());
    // // for (auto o: stress.neuron_layers.back()) std::cout << o << '\t';
    // // std::cout << '\n';

    return 0;
}
