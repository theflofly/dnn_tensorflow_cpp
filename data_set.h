using namespace std;

// Meta data used to normalize the data set. Useful to
// go back and forth between normalized data.
class DataSetMetaData {
friend class DataSet;
private:
  float mean_km;
  float std_km;
  float mean_age;
  float std_age;
  float min_price;
  float max_price;
};

enum class Fuel {
    DIESEL,
    GAZOLINE
};

class DataSet {
public:
  // Construct a data set from the given csv file path.
  DataSet(string path) {
    ReadCSVFile(path);
  }

  // getters
  vector<float>& x() { return x_; }
  vector<float>& y() { return y_; }

  // read the given csv file and complete x_ and y_
  void ReadCSVFile(string path);

  // convert one csv line to a vector of float
  vector<float> ReadCSVLine(string line);

  // normalize a human input using the data set metadata
  initializer_list<float> input(float km, Fuel fuel, float age);

  // convert a price outputted by the DNN to a human price
  float output(float price);
private:
  DataSetMetaData data_set_metadata;
  vector<float> x_;
  vector<float> y_;
};