

#include<bits/stdc++.h>
#include<armadillo>
using namespace std;

// COMMAND TO COMPILE FILE
// g++ arm.cpp -o arm -O1 -larmadillo

// RUN COMMAND
// ./arm

template<typename T>
void get_object_type(T object){
	cout<<typeid(T).name()<<endl;
}

arma::Mat<double> read_csv(string filename, bool skip = true){

	vector< vector<double> > data;

	ifstream fin;

	fin.open(filename);

	vector<double> row;

	string temp,line,word;

	long long int count = 0;

	if(skip){
		getline(fin,line);
	}

	while(fin){

		row.clear();

		getline(fin, line);
			
		if(line.empty()){
			break;
		}

		stringstream s(line);

		try{

			while(getline(s, word,',')){

				row.emplace_back(stod(word));
			}

		}

		catch(...){
			cout<<"KMeans Algorithm Requries Data in Numerical Format Only"<<count<<endl;
			exit(1);
		}

		data.emplace_back(row);

	}

	fin.close();

	arma::Mat<double> dataset(data.size(), data[0].size());

	for(int i=0;i<data.size();i++){
		dataset.row(i) = arma::rowvec(data[i]);
	}

	return dataset;
}

tuple< arma::Row<double>, arma::Row<double> > min_max_of_features(arma::Mat<double>& dataset){

	long long int cols = dataset.n_cols;
	
	arma::Row<double> minimum(cols),maximum(cols);

	tuple<arma::Row<double>, arma::Row<double> > minimum_maximum;

	for(long long int i=0;i<cols;i++){

		minimum[i] = dataset.col(i).min();

		maximum[i] = dataset.col(i).max();

	}

	minimum_maximum = {minimum, maximum};

	return minimum_maximum;

}

arma::Mat<double> initialize(arma::Mat<double> &dataset, long long int clusters
	,arma::Row<double> minimum, arma::Row<double> maximum){

	arma::Mat<double> means(clusters, dataset.n_cols);

	means.fill(0);

	long long int rows = dataset.n_rows, cols = dataset.n_cols,i,j;

	random_device rd;

	mt19937 gen(rd());

	for(i=0;i<clusters;i++){

		for(j=0;j<cols;j++){

			uniform_real_distribution<> uniform_values(minimum[j] + 1, maximum[j] - 1);

			means.row(i).col(j) = uniform_values(gen);

		}

	}

	return means;
}

arma::Row<double> update_means(long long int size_of_cluster, arma::Row<double> &mean, arma::Row<double> &feature){

	long long int i;

	double value;

	for(i=0;i<mean.n_cols;i++){

		value = mean[i];

		value = (value)*(size_of_cluster-1)+feature[i];

		value /= size_of_cluster;

		mean[i] = value;

	}

	return mean;

}

double euclidean_distance(arma::Row<double> &mean, arma::Row<double> &feature){

	double sum = 0;

	long long int i,cols = mean.n_cols;

	for(i=0;i<cols;i++){
		sum	 += pow(feature[i] - mean[i],2);
	}

	return sqrt(sum);
}

long long int identify_cluster(arma::Mat<double> &means, arma::Row<double> &feature){

	long long int index = -1,i;

	double temp = DBL_MAX,distance;

	for(i=0;i<means.n_rows;i++){

		arma::Row<double> mean = means.row(i);

		distance = euclidean_distance(mean, feature);

		if(distance < temp){
			temp = distance;
			index = i;
		}

	}

	return index;
}

void find_clusters(arma::Mat<double> &dataset, arma::Mat<double> &means){

	map<long long int, vector< arma::Row<double> > > cluster_map;

	long long int i,rows = dataset.n_rows;

	for(i=0;i<rows;i++){

		arma::Row<double> dataset_row = dataset.row(i);

		long long int key = identify_cluster(means,dataset_row);

		cluster_map[key].emplace_back(dataset_row);

	}

	for(auto itr : cluster_map){

		cout<<"CLUSTER "<<itr.first<<endl;

		vector< arma::Row<double> > values;

		for(auto x : itr.second){

			cout<<x<<endl;

		}

	}

}

arma::Mat<double> KMEANS(arma::Mat<double> &dataset, long long int clusters,long long int max_iterations = 10000){

	long long int cols = dataset.n_cols,index,size_of_cluster;

	arma::Row<double> minimum(cols),maximum(cols);

	tie(minimum, maximum) = min_max_of_features(dataset);

	arma::Mat<double> means(clusters, cols);

	means = initialize(dataset, clusters, minimum, maximum);

	arma::Row<double> cluster_size(clusters), belongsTo(dataset.n_rows);

	cluster_size.fill(0);
	belongsTo.fill(0);

	for(long long int i = 1;i<=max_iterations;i++){

		bool flag = true;

		for(long long int j = 0;j<dataset.n_rows;j++){

			arma::Row<double> feature_row = dataset.row(j);

			index = identify_cluster(means,feature_row);

			cluster_size[index]++;

			size_of_cluster = cluster_size[index];

			arma::Row<double> mean_row = means.row(index);

			means.row(index) = update_means(size_of_cluster,mean_row,feature_row);

			if(index != belongsTo[j]){
				flag = false;
			}

		}

		if(flag){
			break;
		}

	}

	return means;
}

int main(){

	arma::Mat<double> dataset = read_csv("iris.csv");

	arma::Mat<double> means = KMEANS(dataset,5);

	cout<<"Data Points"<<endl<<endl;

	find_clusters(dataset, means);

	arma::Mat<double> predict = { {1.00,1.5000,2.3000,4.780} , {6.15,3.45,5.48,2.45} };

	cout<<"Prediction"<<endl<<endl;

	find_clusters(predict, means);

	return 0;

}
