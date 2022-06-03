import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.EM;
import weka.clusterers.MakeDensityBasedClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ClusteringWeka {
    String datasetFilepath;

    public ClusteringWeka(String datasetFilepath) throws Exception {
        this.datasetFilepath = datasetFilepath;

    }

    public Clusterer getClusters(int clusterType) throws Exception {
        Clusterer model;
        if (clusterType == 0) model = new MakeDensityBasedClusterer();
        else if (clusterType == 1) model = new EM();
        else {
            SimpleKMeans m = new SimpleKMeans();
            m.setNumClusters(clusterType);
            model = m;
        }
        return model;

    }

    public void performClustering() throws Exception {
        System.out.println("-----------------------Performing clustering on file: " + datasetFilepath + "-----------------------------");
        DataSource src = new DataSource(datasetFilepath);
        Instances dt = src.getDataSet();
        for (int i = 0; i < 7; i++) {
            Clusterer model = getClusters(i);
            model.buildClusterer(dt);
            ClusterEvaluation e = new ClusterEvaluation();
            e.setClusterer(model);
            e.evaluateClusterer(dt);
            System.out.println(e.clusterResultsToString());
            System.out.println(e.getLogLikelihood());
        }

    }

    public static void main(String[] args) throws Exception {
        ClusteringWeka c1 = new ClusteringWeka("iris.arff");
        ClusteringWeka c2 = new ClusteringWeka("grades.arff");
        ClusteringWeka c3 = new ClusteringWeka("labor.arff");
        c1.performClustering();
        c2.performClustering();
        c3.performClustering();

    }
}
