import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
//import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.evaluation.classification.Evaluation;

import java.io.File;
import java.io.IOException;

public class irisClassification {

    private static final int FEATURES_COUNT = 4;
    private static final int CLASSES_COUNT = 3;

    public static void main(String[] args) throws IOException {

        BasicConfigurator.configure();
        testMnist();
//        testModels();

    }

    private static void testMnist() throws IOException {
         /*MultiLayerConfiguration cfg1 =
                new NeuralNetConfiguration.Builder()
                        .weightInit(WeightInit.UNIFORM)
                        .list()
                        .layer(0, new DenseLayer.Builder()
                                .activation(Activation.SIGMOID)
                                .nIn(784)
                                .nOut(794)
                                .build())
                        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                .activation(Activation.SIGMOID)
                                .nIn(794)
                                .nOut(10)
                                .build())
                        .build();
                        */

        final int HEIGHT = 28;
        final int WIDTH = 28;
        final int N_OUTCOMES = 10;
        DataSetIterator train =
                new MnistDataSetIterator(100, 60000, true);
        DataSetIterator test =
                new MnistDataSetIterator(100, 10000, true);

        int channels = 1;
        MultiLayerConfiguration cfg = new NeuralNetConfiguration.Builder()
                .seed(123)
                .l2(0.0005) // ridge regression value
                .updater(new Nesterovs(0.006, 0.9))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1) // nIn need not specified in later layers
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(N_OUTCOMES)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, channels)) // InputType.convolutional for normal image
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(cfg);
        model.init();
        for (int i = 0; i < 100; i++) {
            System.out.print(".");
            model.fit(train);
        }
        Evaluation eval = new Evaluation(10);
        while (test.hasNext()) {
            DataSet testMnist = test.next();
            INDArray predict2 = model.output(testMnist.getFeatures());
            eval.eval(testMnist.getLabels(), predict2);
        }
        System.out.println(eval.stats());

        model.save(new File("./mnist_model"));
    }
    /*
    private static void testModels() {
        try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader.initialize(new FileSplit(
                    new ClassPathResource("iris.data").getFile()));

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, FEATURES_COUNT,
                    CLASSES_COUNT);
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

            createAndEvalFromConfig(config1, trainingData, testingData,"model1");
            createAndEvalFromConfig(config2, trainingData, testingData,"model2");
            createAndEvalFromConfig(config3, trainingData, testingData,"model3");

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

        int epochs = 100;
        for (int i = 0; i <= epochs; i++) {
            model.fit(trainingData);
        }


        INDArray output = model.output(testData.getFeatureMatrix());
        Evaluation eval = new Evaluation(CLASSES_COUNT);
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());
        System.out.println(eval.confusionToString());
        if (filename != null)
            model.save(new File(filename));

    }*/


}
