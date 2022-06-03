import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.IOException;

public class AstroseismologyNeuralNet {

    private static final int FEATURES_COUNT = 3;
    private static final int CLASSES_COUNT = 2;

    public static void main(String[] args) {

        BasicConfigurator.configure();

        testModels();

    }

    private static void testModels() {
        try (RecordReader recordReader = new CSVRecordReader(1, ',')) {
            recordReader.initialize(new FileSplit(
                    new ClassPathResource("data.csv").getFile()));

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, 0, 2);
            DataSet allData = iterator.next();
            allData.shuffle(123);

            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(allData);
            normalizer.transform(allData);

            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testingData = testAndTrain.getTest();

            MultiLayerConfiguration config1 = new NeuralNetConfiguration.Builder()
                    .activation(Activation.TANH)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Nesterovs(0.1, 0.9))
                    .l2(0.0001)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(3).build())
                    .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                    .layer(2, new OutputLayer.Builder(
                            LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX)
                            .nIn(3).nOut(CLASSES_COUNT).build())
                    .backprop(true).pretrain(false)
                    .build();

            MultiLayerConfiguration config2 = new NeuralNetConfiguration.Builder()
                    .activation(Activation.TANH)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Nadam())
                    .l2(0.0001)
                    .list()
                    .layer(0, new DenseLayer.Builder()
                            .nIn(FEATURES_COUNT)
                            .nOut(3)
                            .build())
                    .layer(1, new DenseLayer.Builder()
                            .nIn(3)
                            .nOut(3)
                            .build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX)
                            .nIn(3)
                            .nOut(CLASSES_COUNT)
                            .build())
                    .backprop(true).pretrain(false)
                    .build();

            MultiLayerConfiguration config3 = new NeuralNetConfiguration.Builder()
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.UNIFORM)
                    .updater(new Nesterovs(0.1, 0.9))
                    .list()
                    .layer(0, new DenseLayer.Builder()
                            .nIn(FEATURES_COUNT)
                            .nOut(3)
                            .build())
                    .layer(1, new DenseLayer.Builder()
                            .nIn(3)
                            .nOut(3)
                            .build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                            .nIn(3)
                            .nOut(CLASSES_COUNT)
                            .build())
                    .backprop(true).pretrain(false)
                    .build();


            createAndEvalFromConfig(config1, trainingData, testingData, "model1");
            createAndEvalFromConfig(config2, trainingData, testingData, "model2");
            createAndEvalFromConfig(config3, trainingData, testingData, "model3");


        } catch (Exception e) {
            Thread.dumpStack();
            new Exception("Stack trace").printStackTrace();
            System.out.println("Error: " + e.getLocalizedMessage());
        }
    }

    private static void createAndEvalFromConfig(MultiLayerConfiguration config, DataSet trainingData, DataSet testData, String filename) throws IOException {
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        int epochs = 200;
        for (int i = 0; i <= epochs; i++) {
            model.fit(trainingData);
        }


        INDArray output = model.output(testData.getFeatureMatrix());
        Evaluation eval = new Evaluation(CLASSES_COUNT);
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());
        System.out.println(eval.confusionToString());


        ModelSerializer.writeModel(model, "src/main/resources/models/"+filename+".bin", true);



    }


}
