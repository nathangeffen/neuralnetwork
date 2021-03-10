/**
 * @file dp.hh
 * Declares the data processing functions and classes used by the
 * Net class, including a the dp namespace and a simple CSV file manager class.
 */

#ifndef NN_CSV_HH
#define NN_CSV_HH

#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

#include "nn_matrix.hh"

namespace nn {

    /**
     * Simple CSV file management class. In general read the file into
     * a CSV<std::string> class and then convert to CSV<float>  for processing in
     * neural net.
     */
    template <typename T>
    class CSV {
    public:
        std::vector< std::string > header; ///< Stores the CSV file header
        std::vector< std::vector<T> > rows; ///< Stores the CSV rows
    };

    CSV<std::string> read_csv(std::istream & is, char delim=',', bool header=false);
    void convert_csv_col_labels_to_ints(nn::CSV<std::string> & csv, size_t col);

    template <typename T>
    void convert_csv_types(nn::CSV<std::string> & from, nn::CSV<T> & to) {
        to.rows = {};
        for (auto &row: from.rows) {
            std::vector<T> vec(row.size());
            to.rows.push_back(vec);
        }
        to.header = from.header;
        for (size_t i = 0; i < from.rows.size(); i++) {
            for (size_t j = 0; j < from.rows[i].size(); j++) {
                std::stringstream st(from.rows[i][j]);
                st >> to.rows[i][j];
            }
        }
    }

    template <typename T, class URNG>
    static void shuffle_patterns_targets(std::vector< std::vector<T> >  &patterns,
                                         std::vector< std::vector<T> >  &targets,
                                         URNG&& rng)
    {
        std::vector<size_t> indices(patterns.size());
        for (size_t i = 0; i < indices.size(); i++) indices[i] = i;
        shuffle(indices.begin(), indices.end(), rng);
        for (size_t i = 0; i < indices.size(); i++) {
            std::swap(patterns[i], patterns[indices[i]]);
            std::swap(targets[i], targets[indices[i]]);
        }
    }



    /**
     * Converts a CSV matrix of numbers (usually floats) to patterns and targets
     * in preparation for training and evaluating a neural net on category data.
     *
     * @param csv The CSV object to convert
     * @param patterns The function stores the patterns in this object
     * @param targets The function stores the targets in this object
     * @param categories The number of categories (each target will have an
     * an element for each category)
     * @param rng A uniform random number generator
     */
    template <typename T, class URNG>
    void convert_csv_patterns_targets(const nn::CSV<T> &csv,
                                      std::vector< std::vector<T> >  &patterns,
                                      std::vector< std::vector<T> >  &targets,
                                      unsigned categories = 1,
                                      URNG&& rng=std::mt19937()) {
        for (size_t i = 0; i < csv.rows.size(); i++) {
            std::vector<T> vec_p, vec_t(categories, (T) 0);
            for (size_t j = 0; j < csv.rows[i].size() - 1; j++) {
                vec_p.push_back(csv.rows[i][j]);
            }
            patterns.push_back(vec_p);
            size_t k = csv.rows[i].back();
            vec_t[k] = (T) 1;
            targets.push_back(vec_t);
        }
        shuffle_patterns_targets(patterns, targets, rng);
    }

    template <typename T, class URNG>
    void csv_to_patterns_and_targets(
        CSV<std::string> &csv,
        std::vector< std::vector<T> >  &patterns,
        std::vector< std::vector<T> >  &targets,
        bool convert_last_column = false,
        const std::vector< size_t > &columns_to_convert = {},
        URNG&& rng=std::mt19937()) {
        if (convert_last_column) {
            convert_csv_col_labels_to_ints(csv, csv.rows[0].size() - 1);
        }
        for (auto & index: columns_to_convert) {
            convert_csv_col_labels_to_ints(csv, index);
        }
        nn::CSV<float> csv_f;
        nn::convert_csv_types(csv, csv_f);
        unsigned int max_category = 0;
        for (auto & row: csv_f.rows) {
            if (row.back() > max_category) max_category = row.back();
        }
        nn::convert_csv_patterns_targets(csv_f, patterns, targets, max_category+1,
                                         rng);
    }

    template <typename T, class URNG>
    void csv_stream_to_patterns_and_targets(
        std::ifstream &input,
        std::vector< std::vector<T> >  &patterns,
        std::vector< std::vector<T> >  &targets,
        bool convert_last_column = false,
        const std::vector< size_t > &columns_to_convert = {},
        URNG&& rng=std::mt19937()) {
        CSV<std::string> csv = nn::read_csv(input);
        csv_to_patterns_and_targets(csv, patterns, targets,
                                    convert_last_column,
                                    columns_to_convert, rng);
    }

    template <typename T, class URNG>
    void csv_file_to_patterns_and_targets(
        const char *filename,
        std::vector< std::vector<T> >  &patterns,
        std::vector< std::vector<T> >  &targets,
        bool convert_last_column = false,
        const std::vector< size_t > &columns_to_convert = {},
        URNG&& rng=std::mt19937())
    {
        std::ifstream input(filename, std::ifstream::in);
        csv_stream_to_patterns_and_targets(input, patterns, targets,
                                           convert_last_column,
                                           columns_to_convert, rng);
    }

    template <typename T>
    nn::Matrix<T>
    csv_to_matrix(nn::CSV<std::string> & csv) {
        nn::Matrix<T> matrix(csv.rows.size(), csv.rows[0].size());
        for (size_t i = 0; i < csv.rows.size(); i++) {
            for (size_t j = 0; j < csv.rows[i].size(); j++) {
                T t;
                std::stringstream st(csv.rows[i][j]);
                st >> t;
                matrix[i][j] = t;
            }
        }
        return matrix;
    }

    void rtrim(std::string &s);
    void ltrim(std::string &s);
    void trim(std::string &s);
}


#endif
