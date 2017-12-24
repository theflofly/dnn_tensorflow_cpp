#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include "data_set.h"

using namespace std;

void DataSet::ReadCSVFile(string path) {
  ifstream file(path);
  stringstream buffer;
  buffer << file.rdbuf();
  string line;
  vector<string> lines;
  while(getline(buffer, line, '\n')) {
    lines.push_back(line);
  }

  // the first line contains the metadata
  vector<float> metadata = ReadCSVLine(lines[0]);

  data_set_metadata.mean_km = metadata[0];
  data_set_metadata.std_km = metadata[1];
  data_set_metadata.mean_age = metadata[2];
  data_set_metadata.std_age = metadata[3];
  data_set_metadata.min_price = metadata[4];
  data_set_metadata.max_price = metadata[5];
  
  // the other lines contain the features for each car
  for (int i = 2; i < lines.size(); ++i) {
    vector<float> features = ReadCSVLine(lines[i]);
    x_.insert(x_.end(), features.begin(), features.begin() + 3);
    y_.push_back(features[3]);
  }
}

vector<float> DataSet::ReadCSVLine(string line) {
  vector<float> line_data;
  std::stringstream lineStream(line);
  std::string cell;
  while(std::getline(lineStream, cell, ','))
  {
    line_data.push_back(stod(cell));
  }
  return line_data;
}

initializer_list<float> DataSet::input(float km, Fuel fuel, float age) {
  km = (km - data_set_metadata.mean_km) / data_set_metadata.std_km;
  age = (age - data_set_metadata.mean_age) / data_set_metadata.std_age;
  float f = fuel == Fuel::DIESEL ? -1.f : 1.f;
  return {km, f, age};
}

float DataSet::output(float price) {
  return price * (data_set_metadata.max_price - data_set_metadata.min_price) + data_set_metadata.min_price;
}