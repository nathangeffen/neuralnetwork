/**
 * @file dp.cc
 * Defines the data processing methods and functions declared in dp.hh in the dp
 * namespace. Used to support data processing by the Neural Net class declared
 * in nn.hh.
 */

#include <algorithm>
#include <cassert>
#include <cctype>
#include <locale>
#include <unordered_map>

#include "nn_csv.hh"
#include "nn_matrix.hh"
#include "nn_math.hh"

/// Trim from start of string (in place) (Code originally from Stack Exchange)
void nn::ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

/// Trim from end of string (in place) (Code originally from Stack Exchange)
void nn::rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

/// Trim both ends of a string (in place)
void nn::trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}


/**
 * Convert a column of labels in a CSV file to integers. The column is still a
 * string but this function is often a useful prerequisite to converting an
 * entire matrix of strings into floats. Each label is changed to a unique
 * integer.
 *
 * @param csv The CSV matrix of strings with the column to change.
 * @param col The column number (0-indexed) to convert from labels to integers.
 */
void nn::convert_csv_col_labels_to_ints(nn::CSV<std::string> & csv, size_t col)
{
    std::unordered_map<std::string, unsigned int> conversions;

    if (csv.rows.size() == 0) return;
    assert(csv.rows[0].size() > col);
    unsigned int category = 0;

    for (auto &row: csv.rows) {
        if (conversions.find(row[col]) == conversions.end()) {
            conversions[row[col]] = category++;
        }
    }
    for (auto &row: csv.rows) {
        row[col] = std::to_string(conversions[row[col]]);
    }
}


/**
 * Reads a csv file and returns a CSV class.
 * All blank lines are ignored as well as all lines starting with a #.
 * All entries are saved as trimmed strings in the CSV rows
 *
 * @param is The stream to read from
 * @param header True if an only if the CSV file has a header row
 *
 */

nn::CSV<std::string> nn::read_csv(std::istream & is, char delim, bool header)
{
    nn::CSV<std::string> csv;
    std::string line;
    bool header_processed = false;

    while(std::getline(is, line)) {
        nn::trim(line);
        if (line.size() == 0) continue; // Skip blank lines
        if (line[0] == '#') continue; // Skip comment lines

        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> row;

        while(std::getline(lineStream, cell, delim)) {
            trim(cell);
            row.push_back(cell);
        }
        if (!lineStream && cell.empty())
            row.push_back("");
        if (header == true && header_processed == false) {
            csv.header = row;
            header_processed = true;
        } else {
            csv.rows.push_back(row);
        }
    }
    return csv;
}
