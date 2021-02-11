#ifndef DP_HH
#define DP_HH

#include <sstream>
#include <string>
#include <vector>
#include <iostream>

namespace dp {
    template <typename T>
    class CSV {

    public:
        std::vector< std::string > header;
        std::vector< std::vector<T> > rows;
    };

    CSV<std::string> read_csv(std::istream & is, char delim=',', bool header=false);
    void convert_csv_col_labels_to_ints(dp::CSV<std::string> & csv, size_t col);

    template <typename T>
    void convert_csv_types(dp::CSV<std::string> & from, dp::CSV<T> & to) {
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
    template <typename T>
    void convert_csv_patterns_targets(const dp::CSV<T> &csv,
                                     std::vector< std::vector<T> >  &patterns,
                                     std::vector< std::vector<T> >  &targets,
                                     unsigned categories = 1) {
        for (size_t i = 0; i < csv.rows.size(); i++) {
            std::vector<T> vec_p, vec_t(categories, (T) 0);
            for (size_t j = 0; j < csv.rows[i].size() - 1; j++) {
                vec_p.push_back(csv.rows[i][j]);
            }
            patterns.push_back(vec_p);
            size_t k = csv.rows[i].size() - 1;
            std::cerr << k << ' ' << vec_t.size() << '\n';
            vec_t[k] = (T) 1;
            targets.push_back(vec_t);
        }
    }

    void rtrim(std::string &s);
    void ltrim(std::string &s);
    void trim(std::string &s);
}


#endif
