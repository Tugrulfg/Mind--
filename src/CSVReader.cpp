#include "../include/CSVReader.hpp"

namespace cmind{

    // CSV Dataset constructor
    CSVReader::CSVReader(const std::string& path,  bool with_header): with_header(with_header){
        read(path);
    }

    // Reads the csv file and constructs the dataset
    void CSVReader::read(const std::string& path){
        std::ifstream file(path);
        if(!file.is_open()){
            std::cout << "Unable to open file: " << path << std::endl;
            abort();
        }
        std::string line;

        if(this->with_header){              // Reading the column names if they exists
            std::getline(file, line);
            this->headers = split(line);
        }

        // Reading the data for determining column types and row/column counts
        bool start = true;
        while(std::getline(file, line)){
            this->num_row++;
            if(start){
                start = false;
                std::vector<std::string> data;
                data = split(line);
                this->num_column = data.size();
                size_t pos;
                for(size_t i=0; i<this->num_column; i++){
                    try{
                        std::stoi(data[i], &pos);
                        if(pos == data[i].size())
                            this->column_types.push_back(dtype::INT);
                        else
                            throw std::exception();
                    }
                    catch(const std::exception& e){
                        try{
                            std::stof(data[i], &pos);
                            if(pos == data[i].size())
                                this->column_types.push_back(dtype::FLOAT);
                            else 
                                throw std::exception();
                        }
                        catch(const std::exception& e){
                            if(data[i] == "true" || data[i] == "false")
                                this->column_types.push_back(dtype::BOOL);
                            else
                                this->column_types.push_back(dtype::STR);   
                        }
                    }
                }
            }
        }

        for(dtype type: this->column_types){
            if(type == dtype::BOOL)
                this->columns.push_back(new Tensor<bool>({this->num_row}));
            else if(type == dtype::INT)
                this->columns.push_back(new Tensor<int>({this->num_row}));
            else if(type == dtype::FLOAT)
                this->columns.push_back(new Tensor<float>({this->num_row}));
            else if(type == dtype::STR)
                this->columns.push_back(new Tensor<std::string>({this->num_row}));
        }
        file.close();
        file = std::ifstream(path);
        if(this->with_header)
            std::getline(file, line);
        for(size_t j=0; j<this->num_row; j++){
            std::getline(file, line);
            std::vector<std::string> data;
            data = split(line);

            for(size_t i=0; i<this->num_column; i++){
                if(this->column_types[i] == dtype::BOOL){
                    if(data[i] == "true")
                        ((Tensor<bool>*)(this->columns[i]))[0][j] = true;
                    else
                        ((Tensor<bool>*)(this->columns[i]))[0][j] = false;
                }
                else if(this->column_types[i] == dtype::INT)
                    ((Tensor<int>*)(this->columns[i]))[0][j] = std::stoi(data[i]);
                else if(this->column_types[i] == dtype::FLOAT)
                    ((Tensor<float>*)(this->columns[i]))[0][j] = std::stof(data[i]);
                else if(this->column_types[i] == dtype::STR)
                    ((Tensor<std::string>*)(this->columns[i]))[0][j] = data[i];
            }
        }
        file.close();
    }

    // Splits the line into values according to ','
    std::vector<std::string> CSVReader::split(const std::string& line){
        std::vector<std::string> values;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, ',')) {
            token = trim(token);
            if(!token.empty())
                values.push_back(token);
        }

        return values;
    }

    // Removes the empty spaces from the string
    std::string CSVReader::trim(const std::string& str){
        size_t first = str.find_first_not_of(' ');
        if (std::string::npos == first)
            return std::string();
        size_t last = str.find_last_not_of(' ');
        return str.substr(first, (last - first + 1));
    }

    // Returns the column names
    const std::vector<std::string>& CSVReader::get_headers()const{
        return this->headers;
    }

    // Returns the shape of the dataset
    Shape CSVReader::shape()const{
        return Shape({this->num_row, this->num_column});
    }

    // Returns the data at the given index
    std::tuple<const void*, dtype> CSVReader::at(const size_t row, const size_t col)const{
        if(row >= this->num_row || col >= this->num_column){
            std::cerr << "Dataset: Index out of bounds: " << row << " " << col << std::endl;
            abort();
        }
        dtype type = this->column_types[col];
        if(type == dtype::STR){
            Tensor<std::string>& tensor = *(static_cast<Tensor<std::string>*>(columns[col]));
            return std::make_tuple(tensor[row].data(), type);
        }
        else if(type == dtype::BOOL){
            Tensor<bool>& tensor = *(static_cast<Tensor<bool>*>(columns[col]));
            return std::make_tuple(tensor[row].data(), type);
        }
        else if(type == dtype::INT){
            Tensor<int>& tensor = *(static_cast<Tensor<int>*>(columns[col]));
            return std::make_tuple(tensor[row].data(), type);
        }
        else if(type == dtype::FLOAT){
            Tensor<float>& tensor = *(static_cast<Tensor<float>*>(columns[col]));
            return std::make_tuple(tensor[row].data(), type);
        }
        return std::make_tuple(nullptr, type);
    }

    // Sets the data at the given index
    void CSVReader::set(const size_t row, const size_t col, void* data){
        if(row >= this->num_row || col >= this->num_column){
            std::cerr << "Dataset: Index out of bounds: " << row << " " << col << std::endl;
            abort();
        }
        dtype type = this->column_types[col];
        if(type == dtype::STR){
            Tensor<std::string>& tensor = *(static_cast<Tensor<std::string>*>(columns[col]));
            tensor[row] = *(static_cast<std::string*>(data));
        }
        else if(type == dtype::BOOL){
            Tensor<bool>& tensor = *(static_cast<Tensor<bool>*>(columns[col]));
            tensor[row] = *(static_cast<bool*>(data));
        }
        else if(type == dtype::INT){
            Tensor<int>& tensor = *(static_cast<Tensor<int>*>(columns[col]));
            tensor[row] = *(static_cast<int*>(data));
        }
        else if(type == dtype::FLOAT){
            Tensor<float>& tensor = *(static_cast<Tensor<float>*>(columns[col]));
            tensor[row] = *(static_cast<float*>(data));
        }
    }

    // Accessing a column with column name: r-value
    std::tuple<const void*, dtype> CSVReader::operator[](const std::string& col)const{
        for(size_t i=0; i<this->num_column; i++)
            if(this->headers[i] == col)
                return std::make_tuple(this->columns[i], this->column_types[i]);
        std::cerr << "Column not found: " << col << std::endl;
        abort();
    }

    // Accessing a column with column name: l-value
    std::tuple<void* , dtype> CSVReader::operator[](const std::string& col){
        for(size_t i=0; i<this->num_column; i++)
            if(this->headers[i] == col)
                return std::make_tuple(this->columns[i], this->column_types[i]);
        std::cerr << "Column not found: " << col << std::endl;
        abort();
    }

    // Accessing a column with index: r-value
    std::tuple<void*, dtype> CSVReader::operator[](const size_t idx)const{
        if(idx >= this->num_column){
            std::cerr << "Column index out of bounds: " << idx << std::endl;
            abort();
        }
        return std::make_tuple(this->columns[idx], this->column_types[idx]);
    }

    // Accessing a column with index: l-value
    std::tuple<void* , dtype> CSVReader::operator[](const size_t idx){
        if(idx >= this->num_column){
            std::cerr << "Column index out of bounds: " << idx << std::endl;
            abort();
        }
        return std::make_tuple(this->columns[idx], this->column_types[idx]);
    }

    CSVReader::~CSVReader(){
        for(size_t i=0; i<this->num_column; i++){
            std::cout << "Deleting column: " << this->headers[i] << std::endl;
            if(this->column_types[i] == dtype::STR)
                delete static_cast<Tensor<std::string>*>(this->columns[i]);
            else if(this->column_types[i] == dtype::BOOL)
                delete static_cast<Tensor<bool>*>(this->columns[i]);
            else if(this->column_types[i] == dtype::INT)
                delete static_cast<Tensor<int>*>(this->columns[i]);
            else if(this->column_types[i] == dtype::FLOAT)
                delete static_cast<Tensor<float>*>(this->columns[i]);
        }
    }

    // Overloading the << operator
    std::ostream& operator<<(std::ostream& os, const CSVReader& dataset){
        os << std::endl << "------------------------------------------------------------------------------------" << std::endl;

        for(size_t i=0; i<dataset.num_column; i++)
            os << dataset.headers[i] << "\t";
        
        os << std::endl << "------------------------------------------------------------------------------------" << std::endl;
        for(size_t i=0; i<dataset.num_row; i++){
            for(size_t j=0; j<dataset.num_column; j++){
                if(dataset.column_types[j] == dtype::STR){
                    os << ((Tensor<std::string>*)(dataset.columns[j]))[0][i] << "\t";
                }
                else if(dataset.column_types[j] == dtype::INT)
                    os << ((Tensor<int>*)(dataset.columns[j]))[0][i] << "\t";
                else if(dataset.column_types[j] == dtype::FLOAT)
                    os << ((Tensor<float>*)(dataset.columns[j]))[0][i] << "\t";
                else if(dataset.column_types[j] == dtype::BOOL)
                    os << ((Tensor<bool>*)(dataset.columns[j]))[0][i] << "\t";
            }
            os << std::endl;
        }
        os << "------------------------------------------------------------------------------------" << std::endl << std::endl;
        return os;
    }
}