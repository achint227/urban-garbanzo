import java.util.Arrays;

public class Evaluation {
    double accuracy = 0, precision = 0, recall = 0, f1 = 0;
    double metricLabel;

    public Evaluation(double[][] knownOutput, double[][] testOutput, double metricLabel) {
        this.metricLabel = metricLabel;
        double[] metricArray = new double[]{metricLabel};
        double tp = 0, tn = 0, fp = 0, fn = 0;

        for (int i = 0; i < testOutput.length; i++) {
            if (Arrays.equals(knownOutput[i], metricArray)) {
                if (Arrays.equals(testOutput[i], metricArray)) tp++;
                else fn++;

            } else {

                if (Arrays.equals(testOutput[i], metricArray)) fp++;
                else tn++;
            }


        }
        accuracy = tp + tn / (tp + tn + fp + fn);
        precision = tp == 0 ? 0 : tp / (tp + fp);
        recall = tp == 0 ? 0 : tp / (tp + fn);
        f1 = precision == 0 || recall == 0 ? 0 : 2 * precision * recall / (precision + recall);

    }

    public void print() {
        System.out.println("Evaluation Metrics for class: " + metricLabel);
        System.out.println("Accuracy:\t" + accuracy);
        System.out.println("Precision:\t" + precision);
        System.out.println("Recall:\t" + recall);
        System.out.println("F1 score:\t" + f1);

    }
}
