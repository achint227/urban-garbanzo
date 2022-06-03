import cc.mallet.pipe.*;
import cc.mallet.pipe.iterator.FileIterator;
import cc.mallet.topics.MarginalProbEstimator;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.types.InstanceList;
import cc.mallet.util.CharSequenceLexer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;


public class TopicModelling {
    String dataPathFolder;

    public TopicModelling(String dataPathFolder) {
        this.dataPathFolder = dataPathFolder;
    }

    public InstanceList[] createInstances(double train, double test) {
        File[] directories = new File[1];
        directories[0] = new File("src/main/resources/data/" + dataPathFolder);
        FileIterator it = new FileIterator(
                directories,
                new TextFileFilter(),
                FileIterator.LAST_DIRECTORY
        );
        ArrayList<Pipe> pipeList = new ArrayList<>();
        pipeList.add(new Input2CharSequence("UTF-8"));
        pipeList.add(new CharSequence2TokenSequence(CharSequenceLexer.LEX_WORD_CLASSES));
        pipeList.add(new TokenSequenceRemoveStopwords(false));
        pipeList.add(new TokenSequenceLowercase());
        pipeList.add(new TokenSequence2FeatureSequence());
        SerialPipes pipeline = new SerialPipes(pipeList);
        InstanceList instances = new InstanceList(pipeline);
        instances.addThruPipe(it);
        return instances.split(new double[]{test, train, 0.0});
    }

    public InstanceList[] createInstances() {
        return createInstances(0.9, 0.1);
    }

    public double runModel(InstanceList[] instanceSplit, int numberOfTopics) throws IOException {
        ParallelTopicModel model = new ParallelTopicModel(numberOfTopics, 0.01, 0.01);
        model.addInstances(instanceSplit[0]);
        model.setNumThreads(8);
        model.setNumIterations(1000);
        model.estimate();
        MarginalProbEstimator est = model.getProbEstimator();
        double ll = est.evaluateLeftToRight(instanceSplit[1], 10, false, null);
        System.out.println("Total Log likelihood: " + ll);
        return ll;
    }


    public static void main(String[] args) throws IOException {
        TopicModelling tmHP = new TopicModelling("hp");
        InstanceList[] instanceHP = tmHP.createInstances();
        double maxLL = Double.NEGATIVE_INFINITY;
        int bestIteration = 0;
        for (int i = 2; i < 15; i++) {
            System.out.println("===================Running topic modelling for " + i + " Topics=======================");
            double ll = tmHP.runModel(instanceHP, i);
            if (ll > maxLL) {
                bestIteration = i;
                maxLL = ll;

            }
            System.out.println("==================================================================================");
        }
        System.out.println("=====================================" + bestIteration + "=============================================");


    }
}
