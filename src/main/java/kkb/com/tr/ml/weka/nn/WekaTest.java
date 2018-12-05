package kkb.com.tr.ml.weka.nn;


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.math.BigDecimal;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.NNge;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.*;
import weka.core.FastVector;
import weka.core.Instances;

public class WekaTest {
    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;
        
        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }
        
        return inputReader;
    }
    
    public static Evaluation simpleClassify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception {
        Evaluation validation = new Evaluation(trainingSet);
        
        model.buildClassifier(trainingSet);
        validation.evaluateModel(model, testingSet);
        
        return validation;
    }
    
    public static double calculateAccuracy(FastVector predictions) {
        double correct = 0;
        
        for (int i = 0; i < predictions.size(); i++) {
            NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
            if (np.predicted() == np.actual()) {
                correct++;
            }
        }
        
        return 100 * correct / predictions.size();
    }



    public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
        Instances[][] split = new Instances[2][numberOfFolds];
        
        for (int i = 0; i < numberOfFolds; i++) {
            split[0][i] = data.trainCV(numberOfFolds, i);
            split[1][i] = data.testCV(numberOfFolds, i);
        }
        
        return split;
    }
    
    public static void main(String[] args) throws Exception {
        // I've commented the code as best I can, at the moment.
        // Comments are denoted by "//" at the beginning of the line.
        
        BufferedReader datafile = readDataFile(System.getProperty("user.dir")+"\\dataset\\New.arff");
        
        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        


        
        // Choose a set of classifiers
        Classifier[] models = {     new J48(),
                                    new PART(),
                                    new DecisionTable(),

                                    new OneR(),
                                    new DecisionStump() ,
                                    new NaiveBayes()
        };
        
        // Run for each classifier model
        for(int j = 0; j < models.length; j++) {

            System.out.println(models[j].getClass().getSimpleName());
            System.out.println("==========================");
            // Collect every group of predictions for current model in a FastVector
            FastVector predictions = new FastVector();
            
            // For each training-testing split pair, train and test the classifier
            for(int i = 0; i < data.numInstances(); i++) {
                Evaluation validation = simpleClassify(models[j], data, data);
                predictions.appendElements(validation.predictions());
                
                // Uncomment to see the summary for each training-testing pair.
                //System.out.println(models[j].toString());
                //display metrics
                NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
                System.out.println(np.actual()+"==" + np.predicted());
            }

            // Calculate overall accuracy of current classifier on all splits


            // Print current classifier's name and accuracy in a complicated, but nice-looking way.
            //System.out.println(models[j].getClass().getSimpleName() + ": " + String.format("%.2f%%", accuracy) + "\n=====================");
        }
        
    }
}