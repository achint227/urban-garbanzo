public class BasicNeuralNetwork {

    static double LEARNING_RATE = 0.7;
    int nodesInputLayer;
    int nodesOutputLayer;
    int nodesHiddenLayer;
    double[] arrayValues;
    double[] arrayWeight;
    double[] arrayBias;

    public BasicNeuralNetwork(int input, int hidden, int output) {
        int totalNodes = input + hidden + output;
        int totalWeights = (input * hidden) + (hidden * output);
        nodesInputLayer = input;
        nodesHiddenLayer = hidden;
        nodesOutputLayer = output;
        arrayValues = new double[totalNodes];
        arrayWeight = new double[totalWeights];
        arrayBias = new double[totalNodes];
        for (int i = 0; i < totalNodes; i++)
            arrayBias[i] = 0.5 - (Math.random());
        for (int i = 0; i < arrayWeight.length; i++)
            arrayWeight[i] = 0.5 - (Math.random());
    }

    private double sigmoid(double x) {
        return 1.0 / (1 + Math.exp(-1.0 * x));
    }

    public double[][] testNeuralNet(double[][] input) {
        double[][] result = new double[input.length][nodesOutputLayer];
        for (int i = 0; i < input.length; i++) {
            result[i] = processLayers(input[i]);
        }
        return result;
    }

    public double[] processLayers(double inputValues[]) {
        double result[] = new double[nodesOutputLayer];
        int currentNode = 0;
        int currentWeight = 0;
        // process input layer
        for (int i = 0; i < nodesInputLayer; i++) {
            arrayValues[currentNode++] = inputValues[i];
        }
        // process hidden layer
        for (int i = 0; i < nodesHiddenLayer; i++) {
            double sum = arrayBias[currentNode];
            for (int j = 0; j < nodesInputLayer; j++) {
                sum += arrayValues[j] * arrayWeight[currentWeight++];
            }
            arrayValues[currentNode++] = sigmoid(sum);
        }
        // process output layer
        for (int i = 0; i < nodesOutputLayer; i++) {
            double sum = arrayBias[currentNode];
            for (int j = nodesInputLayer; j < nodesInputLayer + nodesHiddenLayer; j++) {
                sum += arrayValues[j] * arrayWeight[currentWeight++];
            }
            arrayValues[currentNode++] = result[i] = sigmoid(sum);
        }
        return result;
    }

    public void forwardPropagation(double inputValues[]) {
        processLayers(inputValues);

    }

    public void trainNeuralNet(double[][] input, double[][] output, double errorThresh) {
        double err2 = 0.0;
        for (int iteration = 0; iteration < 10000; iteration++) {
            double meanSquareError = 0;
            for (int currentInput = 0; currentInput < input.length; currentInput++) {
                // forward
                this.forwardPropagation(input[currentInput]);
                //  error
                final int outputIndex = this.nodesInputLayer + this.nodesHiddenLayer;
                for (int i = outputIndex; i < this.arrayValues.length; i++) {
                    double err = (output[currentInput][i - outputIndex] - this.arrayValues[i]);
                    meanSquareError += Math.pow(err, 2);
                }
                // back
                this.backPropagation(output[currentInput]);
            }
            double err1 = meanSquareError / (this.nodesOutputLayer + this.nodesHiddenLayer);
            err2 = Math.sqrt(err1);
//            System.out.println(iteration + ", " + err2);
            if (err2 < errorThresh) {
                System.out.println("Error value reached below " + errorThresh + " in iteration no." + iteration + ", " + "exiting training");
                break;
            }
        }
        System.out.println("Training finished, final value of error was " + err2);
    }

    public void backPropagation(double[] actual) {
        double[] derivativeTotalWeight = new double[arrayWeight.length];
        double[] derivativeTotalBias = new double[arrayValues.length];
        double[] derivativeError = new double[arrayValues.length];
        double[] error = new double[arrayValues.length];
        int current;
        // outputs
        current = nodesInputLayer * nodesHiddenLayer;
        for (int i = (nodesInputLayer + nodesHiddenLayer); i < arrayValues.length; i++) {
            error[i] = actual[i - (nodesInputLayer + nodesHiddenLayer)] - arrayValues[i];
            derivativeError[i] = error[i] * arrayValues[i] * (1 - arrayValues[i]);
            for (int j = nodesInputLayer; j < (nodesInputLayer + nodesHiddenLayer); j++) {
                derivativeTotalWeight[current] = derivativeTotalWeight[current] + derivativeError[i] * arrayValues[j];
                error[j] = error[j] + arrayWeight[current] * derivativeError[i];
                current++;
            }
            derivativeTotalBias[i] = derivativeTotalBias[i] + derivativeError[i];
        }
        // hidden
        current = 0;
        for (int i = nodesInputLayer; i < (nodesInputLayer + nodesHiddenLayer); i++) {
            derivativeError[i] = error[i] * arrayValues[i] * (1 - arrayValues[i]);
        }
        for (int i = nodesInputLayer; i < (nodesInputLayer + nodesHiddenLayer); i++) {
            for (int j = 0; j < nodesInputLayer; j++) {
                derivativeTotalWeight[current] = derivativeTotalWeight[current] + derivativeError[i] * arrayValues[j];
                error[j] = error[j] + arrayWeight[current] * derivativeError[i];
                current++;
            }
            derivativeTotalBias[i] = derivativeTotalBias[i] + derivativeError[i];
        }
        // update weights
        for (int i = 0; i < arrayWeight.length; i++) {
            arrayWeight[i] = arrayWeight[i] + (LEARNING_RATE * derivativeTotalWeight[i]);
            derivativeTotalWeight[i] = 0;
        }
        // update bias
        for (int i = nodesInputLayer; i < arrayValues.length; i++) {
            arrayBias[i] = arrayBias[i] + (LEARNING_RATE * derivativeTotalBias[i]);
            derivativeTotalBias[i] = 0;
        }
    }

    public static void main(String[] args) {
        double[][] Input = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
        double[][] xorOutput = {{0.0}, {1.0}, {1.0}, {0.0}};
        double[][] andOutput = {{0.0}, {0.0}, {0.0}, {1.0}};
        double[][] orOutput = {{0.0}, {1.0}, {1.0}, {1.0}};
        double errorThreshold = 0.00;

        BasicNeuralNetwork network1 = new BasicNeuralNetwork(2, 3, 1);
        network1.trainNeuralNet(Input, xorOutput, errorThreshold);
        double[][] result1 = network1.testNeuralNet(Input);
        BasicNeuralNetwork network2 = new BasicNeuralNetwork(2, 3, 1);
        network2.trainNeuralNet(Input, andOutput, errorThreshold);
        double[][] result2 = network2.testNeuralNet(Input);
        BasicNeuralNetwork network3 = new BasicNeuralNetwork(2, 3, 1);
        network3.trainNeuralNet(Input, orOutput, errorThreshold);
        double[][] result3 = network3.testNeuralNet(Input);
        Evaluation eval1 = new Evaluation(xorOutput, result1, 1);
        Evaluation eval2 = new Evaluation(andOutput, result2, 1);
        Evaluation eval3 = new Evaluation(orOutput, result3, 1);
        eval1.print();
        eval2.print();
        eval3.print();

    }

}
/*
Training finished, final value of error was 0.015179286253305882
Training finished, final value of error was 0.009159119703136175
Training finished, final value of error was 0.00700001746413265
Evaluation Metrics for class: 1.0
Accuracy:	0.5
Precision:	0.0
Recall:	0.0
F1 score:	0.0
Evaluation Metrics for class: 1.0
Accuracy:	0.75
Precision:	0.0
Recall:	0.0
F1 score:	0.0
Evaluation Metrics for class: 1.0
Accuracy:	0.25
Precision:	0.0
Recall:	0.0
F1 score:	0.0

 */