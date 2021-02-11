#include "dp.hh"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <locale>
#include <unordered_map>
#include <iostream>

// trim from start (in place)
void dp::ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
void dp::rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
void dp::trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}


void dp::convert_csv_col_labels_to_ints(dp::CSV<std::string> & csv, size_t col)
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

dp::CSV<std::string> dp::read_csv(std::istream & is, char delim, bool header)
{
    dp::CSV<std::string> csv;
    std::string line;
    bool header_processed = false;

    while(std::getline(is, line)) {
        dp::trim(line);
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
