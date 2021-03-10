#ifndef NN_MATRIX_HH
#define NN_MATRIX_HH

#include <cassert>
#include <complex>
#include <iterator>
#include <initializer_list>
#include <functional>
#include <vector>

template <typename T> class ColumnIterator;

namespace nn {
    template <typename T> class Column;

    template <typename T>
    class ColumnIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = T;
        using pointer           = T*;
        using reference         = T&;

        ColumnIterator() = default;
        ColumnIterator(Column<T> & column,
                       size_t pos) : column_(column), pos_(pos) {}

        T operator*()  { return column_[pos_]; }
        pointer operator->() { return &column_[pos_]; }

        // Prefix increment
        ColumnIterator& operator++() {
            pos_++;
            return *this;
        }

        // Postfix increment
        ColumnIterator operator++(int) {
            ColumnIterator tmp = *this;
            ++(*this); return tmp;
        }

        friend bool operator== (const ColumnIterator& a,
                                const ColumnIterator& b) {
            return a.pos_ == b.pos_;
        }
        friend bool operator!= (const ColumnIterator& a,
                                const ColumnIterator& b) {
            return a.pos_ != b.pos_;
        }
        friend ptrdiff_t operator- (const ColumnIterator& a,
                                const ColumnIterator& b) {
            return (a.pos_ - b.pos_);
        }

    private:
        Column<T> column_;
        size_t pos_;
    };

    template <typename T=float>
    class Matrix {
        friend class Column<T>;
    public:
        Matrix() : rows_(1) {
            rows_[0] = std::vector<T>(1);
        };

        Matrix(unsigned int rows, unsigned int cols) {
            init(rows, cols);
        };

        Matrix(unsigned int rows, unsigned int cols, T val) {
            init(rows, cols);
            for_each([&](T & cell) {cell = val;});
        };

        Matrix(unsigned int rows, unsigned int cols,
               std::function<T()> initializer) {
            init(rows, cols);
            for_each([&](unsigned i, unsigned j) {
                A[i][j] = initializer();
            });
        };

        Matrix(unsigned int rows, unsigned int cols,
               std::function<void(T&, unsigned, unsigned)> initializer) {
            init(rows, cols);
            for_each([&](T& cell, unsigned i, unsigned j) {
                initializer(cell, i, j);
            });
        };


        Matrix(const std::initializer_list< std::vector<T> > & source) :
            rows_(source) {}

        Matrix(const Matrix<T> &matrix) { rows_ = matrix.rows_; }

        inline auto begin() const { return A.begin(); }
        inline auto end() const { return A.end(); }
        inline auto begin() { return A.begin(); }
        inline auto end() { return A.end(); }
        inline auto rbegin() const { return A.rbegin(); }
        inline auto rend() const { return A.rend(); }
        inline auto rbegin() { return A.rbegin(); }
        inline auto rend() { return A.rend(); }

        Matrix operator=(const Matrix<T> & matrix) {
            Matrix m(matrix);
            return m;
        }

        inline std::vector<T>& operator[](size_t i) const {
            return A[i];
        }

        inline std::vector<T>& at(size_t i) const {
            return A[i];
        }

        inline unsigned int num_rows() const {return rows_.size(); }
        inline unsigned int num_cols() const {return rows_[0].size(); }

        Matrix operator+(const Matrix<T> &B) const {
            assert(B.num_rows() > 0);
            assert(B.num_cols() == num_cols());
            assert(B.num_rows() == num_rows());

            Matrix C(num_rows(), num_cols());

            for (size_t i = 0; i < num_rows(); i++) {
                for (size_t j = 0; j < num_cols(); j++) {
                    C[i][j] = A[i][j] + B[i][j];
                }
            }
            return C;
        }

        Matrix operator-(const Matrix<T> &B) const {
            assert(B.num_rows() > 0);
            assert(B.num_cols() == num_cols());
            assert(B.num_rows() == num_rows());

            Matrix C(num_rows(), num_cols());

            for (size_t i = 0; i < num_rows(); i++) {
                for (size_t j = 0; j < num_cols(); j++) {
                    C[i][j] = A[i][j] - B[i][j];
                }
            }
            return C;
        }

        Matrix operator*(const Matrix<T> &B) const {
            assert(num_rows() > 0 && B.num_rows() > 0);
            assert(num_cols() == B.num_rows());
            Matrix C(num_rows(), B.num_cols());
            for (size_t i = 0; i < num_rows(); i++) {
                for (size_t j = 0; j < B.num_cols(); j++) {
                    for (size_t k = 0; k < num_cols(); k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
            return C;
        }

        inline std::vector<T> & operator[](unsigned int row) const {
            return A[row];
        }

        inline T at(unsigned int row, unsigned int col) const {
            return A[row][col];
        }

        std::vector< std::vector<T> > & data() const {return A;};

        Column<T> column(unsigned int col) const {
            assert(num_rows() > 0);
            assert(col < num_cols());
            return Column<T>(this, col);
        }

        Matrix<T> transpose() const {
            Matrix<T> B(num_cols(), num_rows());
            for_each([&](unsigned row, unsigned col) {
                B[col][row] = A[row][col];
            });
            return B;
        }

        void for_each(std::function<void(T&)> func) {
            for (auto & row: A) {
                for (auto &val: row) {
                    func(val);
                }
            }
        }

        void for_each(std::function<void(unsigned, unsigned)> func) {
            for (unsigned i = 0; i < num_rows(); i++) {
                for (unsigned j = 0; j < num_cols(); j++) {
                    func(i, j);
                }
            }
        }

        void for_each(std::function<void(unsigned, unsigned)> func) const {
            for (unsigned i = 0; i < num_rows(); i++) {
                for (unsigned j = 0; j < num_cols(); j++) {
                    func(i, j);
                }
            }
        }

        void for_each(std::function<void(T&, unsigned, unsigned)> func) {
            for (unsigned i = 0; i < num_rows(); i++) {
                for (unsigned j = 0; j < num_cols(); j++) {
                    func(A[i][j], i, j);
                }
            }
        }

        void for_each(std::function<void(T&, unsigned, unsigned)> func) const {
            for (unsigned i = 0; i < num_rows(); i++) {
                for (unsigned j = 0; j < num_cols(); j++) {
                    func(A[i][j], i, j);
                }
            }
        }


        /**
         * Returns a new matrix that is a submatrix of this one.
         * Note: m.sub_matrix(m.0, m.0, m.num_rows(), m.num_cols()) == m.
         * In other words one less than the last two co-ordinates specifies the
         * boundary of the sub-matrix
         *
         * @param top_row The top-most row to include in the new matrix
         *
         * @param left_col The left-most column to include in the new matrix
         *
         * @param bottom_row The bottom-most row to include in the new matrix
         *
         * @param right_col The right-most column to include in the new matrix
         */
        Matrix<T>
        sub_matrix(unsigned top_row, unsigned left_col,
                   unsigned bottom_row, unsigned right_col) const {
            assert(num_rows() > bottom_row - top_row);
            assert(num_cols() > right_col - left_col);
            Matrix<T> result(bottom_row - top_row, right_col - left_col);
            for (unsigned i = top_row, r = 0; i < bottom_row; i++, r++) {
                for (unsigned j = left_col, c = 0; j < right_col; j++, c++) {
                    result[r][c] = A[i][j];
                }
            }
            return result;
        }

        std::vector<std::pair<T, T> >
        min_max_rows() const {
            std::vector<std::pair<T, T> > result;
            for (auto &p: A) {
                auto min = *std::min_element(p.begin(), p.end());
                auto max = *std::max_element(p.begin(), p.end());
                result.push_back(std::pair<T, T>(min, max));
            }
            return result;
        }

        std::vector<std::pair<T, T> >
        min_max_cols() const {
            Matrix<T> t = this->transpose();
            return t.min_max_rows();
        }

        Matrix<T>
        normalize_rows(T lo, T hi,
                       const std::vector<std::pair<T, T> > & min_max_rows) const {
            Matrix<T> result = *this;
            auto it = min_max_rows.begin();
            for (auto &r: result.rows_) {
                normalize(r.begin(), r.end(),
                          lo, hi, it->first, it->second);
                it++;
            }
            return result;
        }

        Matrix<T>
        normalize_cols(T lo, T hi,
                       const std::vector<std::pair<T, T> > & min_max_cols) const {
            Matrix<T> t = this->transpose();
            auto result = t.normalize_rows(lo, hi, min_max_cols);
            return result.transpose();
        }



    protected:
        std::vector< std::vector<T> > rows_;
        std::vector< std::vector<T> > & A = rows_;
    private:
        void init(unsigned int rows, unsigned int cols) {
            assert(rows > 0 && cols > 0);
            rows_ = std::vector< std::vector<T> >(rows);
            for (auto it = rows_.begin(); it != rows_.end(); it++) {
                *it = std::vector<T>(cols);
            }
        }
    };

    template <typename T=float>
    class Column {
        friend class ColumnIterator<T>;
    public:
        Column(const Matrix<T> *matrix, unsigned int col) :
            matrix_(matrix), col_(col) {};
        T operator[](const size_t index) {
            return (*matrix_)[index][col_];
        }

        T& at(const size_t index) {
            return (*matrix_)[index][col_];
        }

        ColumnIterator<T> begin() {
            return ColumnIterator<T>( *this, 0);
        }
        ColumnIterator<T> end() {
            return ColumnIterator<T>(*this, matrix_->data().size());
        }

        std::vector<T> toVector() {
            std::vector<T> result(end() - begin());
            for (auto it_f = begin(), it_r = result.begin();
                 it_f != end(); it_f++, it_r++) {
                *it_r = *it_f;
            }
            return result;
        }
    private:
        const Matrix<T> * matrix_;
        const unsigned int col_;
    };

    template <typename T=float, typename S>
    Matrix<T> operator*(S scalar, const Matrix<T> &A) {
        Matrix<T> B(A.num_rows(), A.num_cols());
        for (size_t i = 0; i < A.num_rows(); i++) {
            for (size_t j = 0; j < A.num_cols(); j++) {
                B[i][j] = scalar * A[i][j];
            }
        }
        return B;
    }

    template <typename T=float, typename S>
    Matrix<T> operator*(const Matrix<T> &A, S scalar) {
        return scalar * A;
    }

    template <typename T=float>
    std::vector<T> operator*(const Matrix<T> &A, std::vector<T> v) {
        std::vector<T> result(A.num_rows());
        assert(A.num_cols() == v.size());
        unsigned int i = 0, j = 0;
        for(auto it = A.begin(); it != A.end(); it++) {
            for (size_t j = 0; j < v.size(); j++) {
                result[i] += (*it)[j] * v[j];
            }
        }
        return result;
    }

    template <typename T=float>
    Matrix<T> transpose(const Matrix<T> matrix) {
        return matrix.transpose();
    }

    template <typename T=float>
    Matrix<T>
    sub_matrix(Matrix<T> & matrix, unsigned top_row, unsigned left_col,
               unsigned bottom_row, unsigned right_col) {
        return matrix.sub_matrix(top_row, left_col, bottom_row, right_col);
    }

    template <typename T=float>
    Matrix<T> Ident(unsigned rows) {
        return Matrix<T>(rows, rows,
                         [&](T& val, unsigned i, unsigned j) {
                             if (i == j) val = 1; else val = 0;
                         });
    }


    template <typename T=float>
    std::ostream& operator<< (std::ostream& stream,
                              const Matrix<T>& matrix) {
        auto it_r = matrix.data().begin();
        for (; it_r != matrix.data().end(); it_r++) {
            auto it_c = it_r->begin();
            for (; it_c != it_r->end() - 1;
                 it_c++) {
                stream << *it_c << " ";
            }
            stream << *it_c << '\n';
        }
        return stream;
    }

    template <typename T=float>
    std::ostream& operator<< (std::ostream& stream,
                              const std::vector<T>& vector) {
        for (auto &v: vector) std::cout << v << ' ';
        return stream;
    }
}

#endif
