import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;

public class AstroseismologyGUI extends JFrame {
    private JPanel panel1;
    private JButton classifyButton;
    private JSpinner deltaNuSpinner;
    private JSpinner nuMaxSpinner;
    private JSpinner epsilonSpinner;

    private MultiLayerNetwork model;

    public AstroseismologyGUI() throws IOException {
        model = ModelSerializer.restoreMultiLayerNetwork("src/main/resources/models/model_1.bin", false);

        classifyButton.addActionListener(new ActionListener() {
            /**
             * Invoked when an action occurs.
             *
             * @param e the event to be processed
             */
            @Override
            public void actionPerformed(ActionEvent e) {
                double d = (Double) deltaNuSpinner.getValue();
                double n = (Double) nuMaxSpinner.getValue();
                double eps = (Double) epsilonSpinner.getValue();
                String prediction;
                INDArray array = Nd4j.create(new double[]{d, n, eps});
                int[] pre = model.predict(array);

                if (pre[0] == 0) prediction = "RGB";
                else prediction = "HeB";


                JOptionPane.showMessageDialog(null, prediction, "Prediction", JOptionPane.PLAIN_MESSAGE);
            }
        });
    }


    public static void main(String[] args) throws IOException {
        JFrame frame = new AstroseismologyGUI();
        frame.setTitle("Classification in Astroseismology");
        frame.setContentPane(new AstroseismologyGUI().panel1);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setVisible(true);

    }

    private void createUIComponents() {
        SpinnerModel dmodel = new SpinnerNumberModel(2.50000, 2.0, 18.0, .00001);
        SpinnerModel nmodel = new SpinnerNumberModel(18.00000, 17, 240, .00001);
        SpinnerModel emodel = new SpinnerNumberModel(.010, 0.01, 1, .001);
        deltaNuSpinner = new JSpinner(dmodel);
        nuMaxSpinner = new JSpinner(nmodel);
        epsilonSpinner = new JSpinner(emodel);
        JComponent deditor = new JSpinner.NumberEditor(deltaNuSpinner, "##.#####");
        deltaNuSpinner.setEditor(deditor);
        JComponent neditor = new JSpinner.NumberEditor(nuMaxSpinner, "###.#####");
        nuMaxSpinner.setEditor(neditor);
        JComponent eeditor = new JSpinner.NumberEditor(epsilonSpinner, "#.###");
        epsilonSpinner.setEditor(eeditor);
        // TODO: place custom component creation code here
    }
}
