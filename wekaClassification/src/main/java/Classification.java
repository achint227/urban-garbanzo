import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.File;
import java.util.Random;

public class Classification {
    String filename;
    Instances testInstances;
    Instances trainInstances;

    public Classification(String filename) throws Exception {
        this(filename, false);

    }

    public Classification(String file, boolean arff) throws Exception {
        this.filename = file;
        Instances dt;
        if (arff) {
            DataSource src = new DataSource(filename);
            dt = src.getDataSet();

        } else {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(filename));
            dt = loader.getDataSet();
        }
        dt.randomize(new Random(0));
        int trainingDataSize = (dt.numInstances() * 2) / 3;
        int testDataSize = dt.numInstances() - trainingDataSize;
        trainInstances = new Instances(dt, 0, trainingDataSize);
        testInstances = new Instances(dt, trainingDataSize, testDataSize);
        trainInstances.setClassIndex(trainInstances.numAttributes() - 1);
        testInstances.setClassIndex(testInstances.numAttributes() - 1);
    }

    public void runClassifier(Classifier c) throws Exception {
        c.buildClassifier(trainInstances);
        Evaluation e = new Evaluation(trainInstances);
        e.evaluateModel(c, testInstances);
        System.out.println(e.toSummaryString());
        System.out.println(e.toMatrixString());
        System.out.println(e.toClassDetailsString());
    }

    public void createAndEvaluateClassifiers() throws Exception {
        System.out.println("Running Classification for: " + filename);

        NaiveBayes nb = new NaiveBayes();
        System.out.println("Running Naive Bayes for: " + filename);
        this.runClassifier(nb);

        J48 tree = new J48();
        tree.setUnpruned(true);
        System.out.println("Running Decision tree for: " + filename);
        this.runClassifier(tree);

        RandomForest forest = new RandomForest();
        forest.setNumIterations(100);
        System.out.println("Running Random Forest for: " + filename);
        this.runClassifier(forest);

        System.out.println("Classification ended for: " + filename);

    }

    public static void main(String[] args) throws Exception {
        Classification c1 = new Classification("Acoustic_Extinguisher_Fire_Dataset.arff", true);
        Classification c2 = new Classification("Date_Fruit_Datasets.arff", true);
        Classification c3 = new Classification("Occupancy.csv");
        c1.createAndEvaluateClassifiers();
        c2.createAndEvaluateClassifiers();
        c3.createAndEvaluateClassifiers();
    }

}
