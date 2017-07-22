#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <memory>
#include <vector>
#include <tuple>
#include <map>

#include <iostream>
#include <string.h>
#include <typeinfo>

using namespace std;

namespace py = pybind11;
namespace eig = Eigen;

typedef eig::Matrix<double, 4, 4> Matrix44d;
typedef eig::Matrix<double, 3, 3> Matrix33d;
typedef eig::Matrix<double, 2, 2> Matrix22d;

template<typename Vector>
void from_no_args(Vector& instance){
    new (&instance) Vector();
    instance.setZero();
}

template<typename Vector>
void from_constant(Vector& instance, double constant = 0){
    new (&instance) Vector();
    instance.setOnes();
    instance *= constant;
}


template<typename Vector>
void from_python_list(Vector& instance, vector<double> v){
    if (v.size() != instance.size())
        throw py::index_error("number of elements given, not match Vector size");
    new (&instance) Vector();
    int i = 0;
    for (double vi: v){
        instance[i] = vi;
        i++;
    }
}

template<typename vec_1, typename vec_2>
void from_other(vec_1& instance, vec_2& other){
    new (&instance) vec_1();
    instance.setZero();
    int smaller = instance.size();
    if (smaller > other.size()){
        smaller = other.size();
    }
    for (int i=0; i<smaller; i++){
        instance[i] = other[i];
    }
}

template<typename vec_2>
void from_other_x(eig::VectorXd& instance, vec_2& other){
    new (&instance) eig::VectorXd(other.size());
    for (int i=0; i<other.size(); i++){
        instance[i] = other[i];
    }
}

template<typename Vector>
void vector_from_buffer(Vector& instance, py::buffer b){
    py::buffer_info info = b.request();
    if (info.ndim != 1)
        throw std::runtime_error("Incombatible buffer dimension");
    if (info.size != instance.size())
        throw py::index_error("number of values not matching vector size");
    new (&instance) Vector();
    memcpy(instance.data(), info.ptr, sizeof(double) * instance.size());
}

template<typename Vector>
std::string vector_repr(Vector& self){
    std::ostringstream out_str;
    out_str << "Vector" << self.size() << "d(";
    for (int i=0; i< self.size() - 1; i++)
        out_str << self[i] << ", ";
    out_str << self[self.size()-1] << ")";
    return out_str.str();
}

template<typename Vector>
double vector_getitem(Vector &self, size_t i) {
    if (i >= self.size())
        throw py::index_error("Vector3d index out of range");
    return self(i);
}

template<typename Vector>
Vector cross(Vector &self, Vector &other) {
    return self.cross(other);
}

template<typename Vector>
double dot(Vector &self, Vector &other) {
    if (self.size() != other.size())
        throw runtime_error("sizes not matching");
    return self.dot(other);
}

template<typename Vector>
Vector add(Vector &self, Vector &other) {
    if (self.size() != other.size())
        throw runtime_error("sizes not matching");
    return self + other;
}

template<typename Vector>
Vector radd(Vector &self, int other) {
    return self;
}


template<typename Vector>
Vector sub(Vector &self, Vector &other) {
    if (self.size() != other.size())
        throw runtime_error("sizes not matching");
    return self - other;
}

template<typename Vector>
Vector neg(Vector &self) {
    return - self;
}

template<typename Vector>
Vector pos(Vector &self) {
    return self;
}

template<typename Vector>
Vector mul(Vector &self, double value) {
    return self * value;
}

template<typename Vector>
Vector div(Vector &self, double value) {
    return self / value;
}

void vectorx_from_python_list(eig::VectorXd& instance, vector<double> v){
    if (v.size() < 2)
        throw runtime_error("vector size is too small");
    new (&instance) eig::VectorXd(v.size());
    int i = 0;
    for (double vi: v){
        instance[i] = vi;
        i++;
    }
}

void vectorx_from_buffer(eig::VectorXd& instance, py::buffer b){
    py::buffer_info info = b.request();
    if (info.ndim != 1)
        throw std::runtime_error("Incombatible buffer dimension");
    new (&instance) eig::VectorXd(info.size);
    memcpy(instance.data(), info.ptr, sizeof(double) * instance.size());
}



std::string vectorx_repr(eig::VectorXd& self){
    std::ostringstream out_str;
    out_str << "VectorXd(";
    bool increment = (self.size()>(5*2));
    int end = increment? 5:self.size()-1;
    for (int i=0; i< end; i++)
    {
        out_str << self[i] << ", ";
    }
    if (increment){
        out_str << "....., ";
        for (int i=self.size() - end; i< self.size(); i++)
        {
            out_str << self[i] << ", ";
        }
    }
    out_str << self[self.size()-1] << ")";
    return out_str.str();
}

void vectorx_from_constant(eig::VectorXd& instance, int x, double value=1){
    if (x < 2)
        throw runtime_error("vector size is too small");
    new (& instance) eig::VectorXd(x);
    instance.setOnes();
    instance *= value;
}


template<typename matrix>
void matrix_from_buffer(matrix& self, py::buffer b){
    py::buffer_info info = b.request();
    if (info.ndim != 2)
        throw std::runtime_error("Incombatible buffer dimension");
    if (info.shape[0] != self.rows() && info.shape[2] != self.cols())
        throw runtime_error("incompatible shapes");
    memcpy(self.data(), info.ptr, sizeof(double) * self.size());
}

void matrixx_from_buffer(eig::MatrixXd& instance, py::buffer b){
    py::buffer_info info = b.request();
    if (info.size < 2)
         throw runtime_error("vector size is too small");
    if (info.ndim != 2)
        throw std::runtime_error("Incombatible buffer dimension");
    new (& instance) eig::MatrixXd(info.shape[0], info.shape[1]);
    memcpy(instance.data(), info.ptr, sizeof(double) * instance.size());
}

template<typename matrix>
void matrix_from_constant(matrix &mat, double value)
{
    new (& mat) matrix();
    mat.setOnes();
    mat *= value;
}


void matrixx_from_constant(eig::MatrixXd &mat, int rows, int cols, double value)
{
    new (& mat) eig::MatrixXd(rows, cols);
    mat.setOnes();
    mat *= value;
}


template<typename matrix>
void matrix_from_python_list(matrix &self, vector<vector<double>> list){
    cout << self.rows() << endl;
    if (list.size() != self.rows())
    {
        throw runtime_error("incompatible shapes");
    }
    for (vector<double> l: list){
        if (l.size() != self.cols())
            throw runtime_error("incompatible shapes");
    }
    for (int row_index=0; row_index<self.rows(); row_index++){
        for (int col_index=0; col_index<self.cols(); col_index++){
            self(row_index, col_index) = list[row_index][col_index];
        }
    }
}


void matrixx_from_python_list(eig::MatrixXd &mat, vector<vector<double>> list){
//     check if dimesions is 2
    int rows = list.size();
    int cols = list[0].size();
    for (vector<double> l: list){
        if (l.size() != cols)
            throw runtime_error("incompatible shapes");
    }
    new (& mat) eig::MatrixXd(rows, cols);

    for (int row_index=0; row_index<rows; row_index++){
        for (int col_index=0; col_index<cols; col_index++){
            mat(row_index, col_index) = list[row_index][col_index];
        }
    }
}

void matrixx_from_matrixx(eig::MatrixXd &self, eig::MatrixXd &other){
    new (& self) eig::MatrixXd();
}

template<typename matrix>
double matrixx_getitem(matrix &self, tuple<int, int> t){
    return self(get<0>(t), get<1>(t));
}

template<typename matrix>
eig::VectorXd matrixx_col(matrix &self, int col_index){
    if (col_index >= self.size() || col_index < 0)
        throw runtime_error("row not matrix");
    return self.col(col_index);
}

template<typename matrix>
eig::VectorXd matrixx_row(matrix &self, int row_index){
    if (row_index >= self.row(0).size() || row_index < 0)
        throw runtime_error("row not matrix");
    return self.row(row_index);
}

std::string triplet_repr(eig::Triplet<double>& self){
    std::ostringstream out_str;
    out_str << "triplet(" << self.row() << ", " << self.col() << ", " << self.value() << ")" << endl;
    return out_str.str();
}

void sparse_from_triplets(eig::SparseMatrix<double> &self, vector<eig::Triplet<double>> &trip_list){
    self.setFromTriplets(trip_list.begin(), trip_list.end());
}

template<typename matrix1, typename matrix2>
eig::MatrixXd mat_mul(matrix1 &self, matrix2 &other){
    if (self.cols() != other.rows())
        throw runtime_error("incombatible shapes");
    eig::MatrixXd res(self.rows(), other.cols());
    return self * other;
}

vector<vector<double>> matrixx_as_list(eig::MatrixXd &self){
    vector<vector<double>> out;
    for (int i=0; i<self.rows(); i++)
    {
        vector<double> temp;
        for (int j=0; j<self.cols(); j++)
        {
            temp.push_back(self(i, j));
        }
        out.push_back(temp);
    }
    return out;
}

template<typename t1, typename t2>
eig::VectorXd py_and(t1 & self, t2 & other){
    eig::VectorXd new_vec(self.rows() + other.rows());
    new_vec << self, other;
    return new_vec;
}


void init_paraEigen(py::module &m){
    py::class_<eig::Vector2d, std::shared_ptr<eig::Vector2d>>(m, "vector2")
        .def(py::init<double, double>())
        .def_property("x", [](eig::Vector2d &v){return v.x();},
                           [](eig::Vector2d &v, double x){v.x() = x;})
        .def_property("y", [](eig::Vector2d &v){return v.y();},
                           [](eig::Vector2d &v, double y){v.y() = y;})
        .def("__init__", &from_no_args<eig::Vector2d>)
        .def("__init__", &from_constant<eig::Vector2d>)
        .def("__init__", &from_python_list<eig::Vector2d>)
        .def("__init__", &from_other<eig::Vector2d, eig::Vector2d>)
        .def("__init__", &from_other<eig::Vector2d, eig::Vector3d>)
        .def("__init__", &from_other<eig::Vector2d, eig::Vector4d>)
        .def("__init__", &from_other<eig::Vector2d, eig::VectorXd>)
        .def("__init__", &vector_from_buffer<eig::Vector2d>)
        .def("__repr__", &vector_repr<eig::Vector2d>)
        .def("__getitem__", &vector_getitem<eig::Vector2d>)
        .def("__len__",[](eig::Vector2d &v){return v.size();})
        .def("__abs__",[](eig::Vector2d &v){return v.norm();})
        .def("__add__", &add<eig::Vector2d>)
        .def("__and__", &py_and<eig::Vector2d, eig::VectorXd>)
        .def("__and__", &py_and<eig::Vector2d, eig::Vector2d>)
        .def("__and__", &py_and<eig::Vector2d, eig::Vector3d>)
        .def("__and__", &py_and<eig::Vector2d, eig::Vector4d>)
        .def("__radd__", &radd<eig::Vector2d>)
        .def("__sub__", &sub<eig::Vector2d>)
        .def("__neg__", &neg<eig::Vector2d>)
        .def("__pos__", &pos<eig::Vector2d>)
        .def("__mul__", &mul<eig::Vector2d>)
        .def("__rmul__", &mul<eig::Vector2d>)
        .def("__div__", &div<eig::Vector2d>)
        .def("__truediv__", &div<eig::Vector2d>)
        .def("norm",[](eig::Vector2d &v){return v.norm();})
        .def_property_readonly("normal", [](eig::Vector2d &v){return eig::Vector2d(-v.y(), v.x());})
        .def("normalize", [](eig::Vector2d &v){v.normalize(); return v;})
        .def("dot", &dot<eig::Vector2d>);


    py::class_<eig::Vector3d, std::shared_ptr<eig::Vector3d>>(m, "vector3")
        .def(py::init<double, double, double>())        
        .def_property("x", [](eig::Vector3d &v){return v.x();},
                           [](eig::Vector3d &v, double x){v.x() = x;})
        .def_property("y", [](eig::Vector3d &v){return v.y();},
                           [](eig::Vector3d &v, double y){v.y() = y;})
        .def_property("z", [](eig::Vector3d &v){return v.z();},
                           [](eig::Vector3d &v, double z){v.z() = z;})
        .def("__init__", &from_no_args<eig::Vector3d>)
        .def("__init__", &from_constant<eig::Vector3d>)
        .def("__init__", &from_python_list<eig::Vector3d>)
        .def("__init__", &from_other<eig::Vector3d, eig::Vector2d>)
        .def("__init__", &from_other<eig::Vector3d, eig::Vector3d>)
        .def("__init__", &from_other<eig::Vector3d, eig::Vector4d>)
        .def("__init__", &from_other<eig::Vector3d, eig::VectorXd>)
        .def("__init__", &vector_from_buffer<eig::Vector3d>)
        .def("__repr__", &vector_repr<eig::Vector3d>)
        .def("__getitem__", &vector_getitem<eig::Vector3d>)
        .def("size", [](eig::Vector3d &v){return v.size();})
        .def("__len__",[](eig::Vector3d &v){return v.size();})
        .def("__abs__",[](eig::Vector3d &v){return v.norm();})
        .def("__add__", &add<eig::Vector3d>)
        .def("__and__", &py_and<eig::Vector3d, eig::VectorXd>)
        .def("__and__", &py_and<eig::Vector3d, eig::Vector2d>)
        .def("__and__", &py_and<eig::Vector3d, eig::Vector3d>)
        .def("__and__", &py_and<eig::Vector3d, eig::Vector4d>)
        .def("__radd__", &radd<eig::Vector3d>)
        .def("__sub__", &sub<eig::Vector3d>)
        .def("__neg__", &neg<eig::Vector3d>)
        .def("__pos__", &pos<eig::Vector3d>)
        .def("__mul__", &mul<eig::Vector3d>)
        .def("__rmul__", &mul<eig::Vector3d>)
        .def("__div__", &div<eig::Vector3d>)
        .def("__truediv__", &div<eig::Vector3d>)
        .def("norm",[](eig::Vector3d &v){return v.norm();})
        .def("normalize", [](eig::Vector3d &v){v.normalize(); return v;})
        .def("cross", &cross<eig::Vector3d>)
        .def("dot", &dot<eig::Vector3d>);
        
        
    py::class_<eig::Vector4d, std::shared_ptr<eig::Vector4d>>(m, "vector4")
        .def(py::init<double, double, double, double>())
        .def_property("x", [](eig::Vector4d &v){return v.x();},
                           [](eig::Vector4d &v, double x){v.x() = x;})
        .def_property("y", [](eig::Vector4d &v){return v.y();},
                           [](eig::Vector4d &v, double y){v.y() = y;})
        .def_property("z", [](eig::Vector4d &v){return v.z();},
                           [](eig::Vector4d &v, double z){v.z() = z;})
//         .def_property("w", [](eig::Vector4d &w){return v.w();},
//                            [](eig::Vector4d &w, double w){v.() = w;})
        .def("__init__", &from_no_args<eig::Vector4d>)
        .def("__init__", &from_constant<eig::Vector4d>)
        .def("__init__", &from_python_list<eig::Vector4d>)
        .def("__init__", &from_other<eig::Vector4d, eig::Vector2d>)
        .def("__init__", &from_other<eig::Vector4d, eig::Vector3d>)
        .def("__init__", &from_other<eig::Vector4d, eig::Vector4d>)
        .def("__init__", &from_other<eig::Vector4d, eig::VectorXd>)
        .def("__init__", &vector_from_buffer<eig::Vector4d>)
        .def("__repr__", &vector_repr<eig::Vector4d>)
        .def("__getitem__", &vector_getitem<eig::Vector4d>)
        .def("__len__",[](eig::Vector4d &v){return v.size();})
        .def("__abs__",[](eig::Vector4d &v){return v.norm();})
        .def("__add__", &add<eig::Vector4d>)
        .def("__and__", &py_and<eig::Vector4d, eig::VectorXd>)
        .def("__and__", &py_and<eig::Vector4d, eig::Vector2d>)
        .def("__and__", &py_and<eig::Vector4d, eig::Vector3d>)
        .def("__and__", &py_and<eig::Vector4d, eig::Vector4d>)
        .def("__radd__", &add<eig::Vector4d>)
        .def("__sub__", &sub<eig::Vector4d>)
        .def("__neg__", &neg<eig::Vector4d>)
        .def("__pos__", &pos<eig::Vector4d>)
        .def("__mul__", &mul<eig::Vector4d>)
        .def("__rmul__", &mul<eig::Vector4d>)
        .def("__div__", &div<eig::Vector4d>)
        .def("__truediv__", &div<eig::Vector4d>)
        .def("norm",[](eig::Vector4d &v){return v.norm();})
        .def("normalize", [](eig::Vector4d &v){v.normalize(); return v;})
        .def("dot", &dot<eig::Vector4d>);
        
    py::class_<eig::VectorXd, std::shared_ptr<eig::VectorXd>>(m, "vectorx")
        .def(py::init<>())
        .def("__init__", &vectorx_from_constant, py::arg("x"), py::arg("value")=0)
        .def("__init__", &vectorx_from_python_list)
        .def("__init__", &from_other_x<eig::Vector2d>)
        .def("__init__", &from_other_x<eig::Vector3d>)
        .def("__init__", &from_other_x<eig::Vector4d>)
        .def("__init__", &from_other_x<eig::VectorXd>)
        .def("__init__", &vectorx_from_buffer)
        .def("__repr__", &vectorx_repr)
        .def("__getitem__", &vector_getitem<eig::VectorXd>)
        .def("__len__",[](eig::VectorXd &v){return v.size();})
        .def("__len__",[](eig::VectorXd &v){return v.size();})
        .def("__abs__",[](eig::VectorXd &v){return v.norm();})
        .def("__add__", &add<eig::VectorXd>)
        .def("__and__", &py_and<eig::VectorXd, eig::VectorXd>)
        .def("__and__", &py_and<eig::VectorXd, eig::Vector2d>)
        .def("__and__", &py_and<eig::VectorXd, eig::Vector3d>)
        .def("__and__", &py_and<eig::VectorXd, eig::Vector4d>)
        .def("__sub__", &sub<eig::VectorXd>)
        .def("__neg__", &neg<eig::VectorXd>)
        .def("__pos__", &pos<eig::VectorXd>)
        .def("__mul__", &mul<eig::VectorXd>)
        .def("__rmul__", &mul<eig::VectorXd>)
        .def("__div__", &div<eig::VectorXd>)
        .def("__truediv__", &div<eig::VectorXd>)
        .def("norm",[](eig::VectorXd &v){return v.norm();})
        .def("normalize", [](eig::VectorXd &v){v.normalize(); return v;})
        .def("dot", &dot<eig::VectorXd>);
        
    py::class_<Matrix44d>(m, "matrix44")
        .def("__init__", &matrix_from_python_list<Matrix44d>)
        .def("__init__", &matrix_from_constant<Matrix44d>)
        .def("__init__", &matrix_from_buffer<Matrix44d>)
        .def("__getitem__", &matrixx_getitem<Matrix44d>)
        .def("col", &matrixx_col<Matrix44d>)
        .def("row", &matrixx_row<Matrix44d>)
//         .def("tolist", &matrixx_as_list)
//         .def("__mul__", &mat_mul<eig::MatrixXd, eig::MatrixXd>)
//         .def("__mul__", &mat_mul<eig::MatrixXd, eig::SparseMatrix<double>>)
        ;
    
    py::class_<Matrix33d>(m, "matrix33")
        .def("__init__", &matrix_from_python_list<Matrix33d>)
        .def("__init__", &matrix_from_constant<Matrix33d>)
        .def("__init__", &matrix_from_buffer<Matrix33d>)
        .def("__getitem__", &matrixx_getitem<Matrix33d>)
        .def("col", &matrixx_col<Matrix33d>)
        .def("row", &matrixx_row<Matrix33d>)
//         .def("tolist", &matrixx_as_list)
//         .def("__mul__", &mat_mul<eig::MatrixXd, eig::MatrixXd>)
//         .def("__mul__", &mat_mul<eig::MatrixXd, eig::SparseMatrix<double>>)
        ;
        
    py::class_<Matrix22d>(m, "matrix22")
        .def("__init__", &matrix_from_python_list<Matrix22d>)
        .def("__init__", &matrix_from_constant<Matrix22d>)
        .def("__init__", &matrix_from_buffer<Matrix22d>)
        .def("__getitem__", &matrixx_getitem<Matrix22d>)
        .def("col", &matrixx_col<Matrix22d>)
        .def("row", &matrixx_row<Matrix22d>)
//         .def("tolist", &matrixx_as_list)
//         .def("__mul__", &mat_mul<eig::MatrixXd, eig::MatrixXd>)
//         .def("__mul__", &mat_mul<eig::MatrixXd, eig::SparseMatrix<double>>)
        ;
        
    py::class_<eig::MatrixXd>(m, "matrixx")
        .def("__init__", &matrixx_from_matrixx)
        .def("__init__", &matrixx_from_constant)
        .def("__init__", &matrixx_from_python_list)
        .def("__init__", &matrixx_from_buffer)
        .def("__getitem__", &matrixx_getitem<eig::MatrixXd>)
        .def("__getitem__", &matrixx_row<eig::MatrixXd>)
        .def("col", &matrixx_col<eig::MatrixXd>)
        .def("row", &matrixx_row<eig::MatrixXd>)
        .def("__len__",[](eig::MatrixXd &mat){return mat.rows();})
        .def("tolist", &matrixx_as_list)
        .def("__mul__", &mat_mul<eig::MatrixXd, eig::MatrixXd>)
        .def("__mul__", &mat_mul<eig::MatrixXd, eig::SparseMatrix<double>>)
        .def("__mul__", [](eig::MatrixXd& mat, double c){return eig::MatrixXd(c * mat);})
        .def("__matmul__", &mat_mul<eig::MatrixXd, eig::MatrixXd>)
        .def("__matmul__", &mat_mul<eig::MatrixXd, eig::SparseMatrix<double>>)
        .def("__matmul__", [](eig::MatrixXd& mat, double c){return eig::MatrixXd(c * mat);})
        .def_property_readonly("T", [](eig::MatrixXd& mat){return eig::MatrixXd(mat.transpose());});
        
    py::class_<eig::Triplet<double>>(m, "triplet")
        .def(py::init<int, int, double>())
        .def("__repr__", &triplet_repr);
        
    py::class_<eig::SparseMatrix<double>>(m, "sparse_matrix")
        .def(py::init<int, int>())
        .def("set_from_triplets", &sparse_from_triplets)
        .def("__mul__", &mat_mul<eig::SparseMatrix<double>, eig::MatrixXd>)
        .def("__mul__", &mat_mul<eig::SparseMatrix<double>, eig::SparseMatrix<double>>)
        .def("__matmul__", &mat_mul<eig::SparseMatrix<double>, eig::MatrixXd>)
        .def("__matmul__", &mat_mul<eig::SparseMatrix<double>, eig::SparseMatrix<double>>)
        .def_property_readonly("T", [](eig::SparseMatrix<double> mat)
                                      {return eig::SparseMatrix<double>(mat.transpose());})
        .def("to_matrixx",[](eig::SparseMatrix<double> &self){return eig::MatrixXd(self);});
}

py::module m("paraEigen");
PYBIND11_MODULE(paraEigen, m){
    init_paraEigen(m);
};
