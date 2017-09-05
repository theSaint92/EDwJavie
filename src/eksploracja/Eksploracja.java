package eksploracja;

import java.util.Random;

import weka.attributeSelection.ReliefFAttributeEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.global.GeneticSearch;
import weka.classifiers.bayes.net.search.global.HillClimber;
import weka.classifiers.bayes.net.search.global.RepeatedHillClimber;
import weka.classifiers.bayes.net.search.global.SimulatedAnnealing;
import weka.classifiers.bayes.net.search.global.TAN;
import weka.classifiers.bayes.net.search.global.TabuSearch;
import weka.classifiers.bayes.net.search.local.LAGDHillClimber;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;


public class Eksploracja {

	public static void main(String[] args) throws Exception {
		
		Instances banknoty = DataSource.read("./sampledata/banknote-authentication.arff");
		banknoty.setClassIndex(4);
        
        /*======================================================================
         *====================================================================== 
         * MultiLayer Perceptron
         *======================================================================
         *======================================================================
         */
        
        MultilayerPerceptron per = new MultilayerPerceptron();
        per.setTrainingTime(1000);
        per.setMomentum(0.2);
        per.setLearningRate(0.3);
        per.setHiddenLayers("4");
        per.setValidationSetSize(0);
        //per.setGUI(true);
        for (int i=0; i<per.getOptions().length; i++) {
        	System.out.print(per.getOptions()[i] + " ");
        }
        
        per.buildClassifier(banknoty);
        Evaluation ev = new Evaluation(banknoty);
        
        ev.crossValidateModel(per, banknoty, 10, new Random());
        
        
        System.out.println("\n=====================================================");
        System.out.println("MultiLayer Perceptron");
        System.out.println("=====================================================");
        System.out.println(per.toString());   
        System.out.println(ev.toSummaryString());
        System.out.println(ev.toMatrixString());

        /*======================================================================
         *====================================================================== 
         * Sieæ Bayesowska
         *======================================================================
         *======================================================================
         */
        
        int folds = 10;      
        int runs = 4;
       
        BayesNet BN1 = new BayesNet();
        BN1.buildClassifier(banknoty);
        Evaluation eval_train1 = new Evaluation(banknoty);
        eval_train1.evaluateModel(BN1,banknoty);
        System.out.println("==========Sieæ Bayesowska z domyœlnymi parametrami============");
        System.out.println(eval_train1.toSummaryString());
        System.out.println(eval_train1.toMatrixString());
               
        // perform cross-validation
	    for (int i = 0; i < runs; i++) {
	      // randomize data
	      int seed = i + 1;
	      java.util.Random rand = new java.util.Random(seed);
	      Instances randData = new Instances(banknoty);
	      randData.randomize(rand);
	      if (randData.classAttribute().isNominal())
	        randData.stratify(folds);
	 
	      Evaluation eval = new Evaluation(randData);
	      for (int n = 0; n < folds; n++) {
	        Instances train = randData.trainCV(folds, n);
	        Instances test = randData.testCV(folds, n);
	       
	       
	        BN1.buildClassifier(train);
	        eval.evaluateModel(BN1, test);
	      }
	 
	      // output evaluation
	      System.out.println();
	      System.out.println("=== Sieæ Bayesowska z domyœlnymi parametrami run " + (i+1) + " ===");
	      System.out.println("Classifier: " + BN1.getClass().getName() + " " + Utils.joinOptions(BN1.getOptions()));
	      System.out.println("Dataset: " + banknoty.relationName());
	      System.out.println("Folds: " + folds);
	      System.out.println("Seed: " + seed);
	      System.out.println();
	      System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run " + (i+1) + "===", false)+eval.toMatrixString());
	       
	    }
 
	    
	    
	    
	    
        SearchAlgorithm SA2 = new SimulatedAnnealing();
        BayesNet BN2 = new BayesNet();
        BN2.setSearchAlgorithm(SA2);
        BN2.buildClassifier(banknoty);
        Evaluation eval_train2 = new Evaluation(banknoty);
        eval_train2.evaluateModel(BN2,banknoty);
        System.out.println("==========Sieæ Bayesowska z algorytmem szukania:SimulatedAnnealing==========");
        System.out.println(eval_train2.toSummaryString());
        System.out.println(eval_train2.toMatrixString());
       
        // perform cross-validation
	    for (int i = 0; i < runs; i++) {
	      // randomize data
	      int seed = i + 1;
	      java.util.Random rand = new java.util.Random(seed);
	      Instances randData = new Instances(banknoty);
	      randData.randomize(rand);
	      if (randData.classAttribute().isNominal())
	        randData.stratify(folds);
	 
	      Evaluation eval = new Evaluation(randData);
	      for (int n = 0; n < folds; n++) {
	        Instances train = randData.trainCV(folds, n);
	        Instances test = randData.testCV(folds, n);
	       
	       
	        BN2.buildClassifier(train);
	        eval.evaluateModel(BN2, test);
	      }
	 
	      // output evaluation
	      System.out.println();
	      System.out.println("=== Sieæ Bayesowska z algorytmem szukania:SimulatedAnnealing run " + (i+1) + " ===");
	      System.out.println("Classifier: " + BN2.getClass().getName() + " " + Utils.joinOptions(BN2.getOptions()));
	      System.out.println("Dataset: " + banknoty.relationName());
	      System.out.println("Folds: " + folds);
	      System.out.println("Seed: " + seed);
	      System.out.println();
	      System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run " + (i+1) + "===", false)+eval.toMatrixString());
	       
	    }
       
        SearchAlgorithm SA3 = new TabuSearch();
        BayesNet BN3 = new BayesNet();
        BN3.setSearchAlgorithm(SA3);
        BN3.buildClassifier(banknoty);
        Evaluation eval_train3 = new Evaluation(banknoty);
        eval_train3.evaluateModel(BN3,banknoty);
        System.out.println("============Sieæ Bayesowska z algorytmem szukania:TabuSearch==========");
        System.out.println(eval_train3.toSummaryString());
        System.out.println(eval_train3.toMatrixString());
       
        // perform cross-validation
	    for (int i = 0; i < runs; i++) {
	      // randomize data
	      int seed = i + 1;
	      java.util.Random rand = new java.util.Random(seed);
	      Instances randData = new Instances(banknoty);
	      randData.randomize(rand);
	      if (randData.classAttribute().isNominal())
	        randData.stratify(folds);
	 
	      Evaluation eval = new Evaluation(randData);
	      for (int n = 0; n < folds; n++) {
	        Instances train = randData.trainCV(folds, n);
	        Instances test = randData.testCV(folds, n);
	       
	       
	        BN3.buildClassifier(train);
	        eval.evaluateModel(BN3, test);
	      }
	 
	      // output evaluation
	      System.out.println();
	      System.out.println("=== Sieæ Bayesowska z algorytmem szukania:TabuSearch run " + (i+1) + " ===");
	      System.out.println("Classifier: " + BN3.getClass().getName() + " " + Utils.joinOptions(BN3.getOptions()));
	      System.out.println("Dataset: " + banknoty.relationName());
	      System.out.println("Folds: " + folds);
	      System.out.println("Seed: " + seed);
	      System.out.println();
	      System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run " + (i+1) + "===", false)+eval.toMatrixString());
	       
	    }
       
        SearchAlgorithm SA4 = new TAN();
        BayesNet BN4 = new BayesNet();
        BN4.setSearchAlgorithm(SA4);
        BN4.buildClassifier(banknoty);
        Evaluation eval_train4 = new Evaluation(banknoty);
        eval_train4.evaluateModel(BN4,banknoty);
        System.out.println("==========Sieæ Bayesowska z algorytmem szukania:TAN==========");
        System.out.println(eval_train4.toSummaryString());
        System.out.println(eval_train4.toMatrixString());
       
        // perform cross-validation
	    for (int i = 0; i < runs; i++) {
	      // randomize data
	      int seed = i + 1;
	      java.util.Random rand = new java.util.Random(seed);
	      Instances randData = new Instances(banknoty);
	      randData.randomize(rand);
	      if (randData.classAttribute().isNominal())
	        randData.stratify(folds);
	 
	      Evaluation eval = new Evaluation(randData);
	      for (int n = 0; n < folds; n++) {
	        Instances train = randData.trainCV(folds, n);
	        Instances test = randData.testCV(folds, n);
	       
	       
	        BN4.buildClassifier(train);
	        eval.evaluateModel(BN4, test);
	      }
	 
	      // output evaluation
	      System.out.println();
	      System.out.println("=== Sieæ Bayesowska z algorytmem szukania:TAN run " + (i+1) + " ===");
	      System.out.println("Classifier: " + BN4.getClass().getName() + " " + Utils.joinOptions(BN4.getOptions()));
	      System.out.println("Dataset: " + banknoty.relationName());
	      System.out.println("Folds: " + folds);
	      System.out.println("Seed: " + seed);
	      System.out.println();
	      System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run " + (i+1) + "===", false)+eval.toMatrixString());
	       
	    }
           
        SearchAlgorithm SA5 = new GeneticSearch();
        BayesNet BN5 = new BayesNet();
        BN5.setSearchAlgorithm(SA5);
        BN5.buildClassifier(banknoty);
        Evaluation eval_train5 = new Evaluation(banknoty);
        eval_train5.evaluateModel(BN5,banknoty);
        System.out.println("==========Sieæ Bayesowska z algorytmem szukania:GeneticSearch==========");
        System.out.println(eval_train5.toSummaryString());
        System.out.println(eval_train5.toMatrixString());
       
        // perform cross-validation
	    for (int i = 0; i < runs; i++) {
	      // randomize data
	      int seed = i + 1;
	      java.util.Random rand = new java.util.Random(seed);
	      Instances randData = new Instances(banknoty);
	      randData.randomize(rand);
	      if (randData.classAttribute().isNominal())
	        randData.stratify(folds);
	 
	      Evaluation eval = new Evaluation(randData);
	      for (int n = 0; n < folds; n++) {
	        Instances train = randData.trainCV(folds, n);
	        Instances test = randData.testCV(folds, n);
	       
	       
	        BN5.buildClassifier(train);
	        eval.evaluateModel(BN5, test);
	      }
	 
	      // output evaluation
	      System.out.println();
	      System.out.println("=== Sieæ Bayesowska z algorytmem szukania:GeneticSearch run " + (i+1) + " ===");
	      System.out.println("Classifier: " + BN5.getClass().getName() + " " + Utils.joinOptions(BN5.getOptions()));
	      System.out.println("Dataset: " + banknoty.relationName());
	      System.out.println("Folds: " + folds);
	      System.out.println("Seed: " + seed);
	      System.out.println();
	      System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run " + (i+1) + "===", false)+eval.toMatrixString());
	       
	    }
 
        SearchAlgorithm SA6 = new HillClimber();
        BayesNet BN6 = new BayesNet();
        BN6.setSearchAlgorithm(SA6);
        BN6.buildClassifier(banknoty);
        Evaluation eval_train6 = new Evaluation(banknoty);
        eval_train6.evaluateModel(BN6,banknoty);
        System.out.println("==========Sieæ Bayesowska z algorytmem szukania:HillClimber==========");
        System.out.println(eval_train6.toSummaryString());
        System.out.println(eval_train6.toMatrixString());
       
        // perform cross-validation
	    for (int i = 0; i < runs; i++) {
	      // randomize data
	      int seed = i + 1;
	      java.util.Random rand = new java.util.Random(seed);
	      Instances randData = new Instances(banknoty);
	      randData.randomize(rand);
	      if (randData.classAttribute().isNominal())
	        randData.stratify(folds);
	 
	      Evaluation eval = new Evaluation(randData);
	      for (int n = 0; n < folds; n++) {
	        Instances train = randData.trainCV(folds, n);
	        Instances test = randData.testCV(folds, n);
	       
	       
	        BN6.buildClassifier(train);
	        eval.evaluateModel(BN6, test);
	      }
	 
	      // output evaluation
	      System.out.println();
	      System.out.println("=== Sieæ Bayesowska z algorytmem szukania:HillClimber run " + (i+1) + " ===");
	      System.out.println("Classifier: " + BN6.getClass().getName() + " " + Utils.joinOptions(BN6.getOptions()));
	      System.out.println("Dataset: " + banknoty.relationName());
	      System.out.println("Folds: " + folds);
	      System.out.println("Seed: " + seed);
	      System.out.println();
	      System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run " + (i+1) + "===", false)+eval.toMatrixString());
	       
	    }
       
        SearchAlgorithm SA7 = new LAGDHillClimber();
        BayesNet BN7 = new BayesNet();
        BN7.setSearchAlgorithm(SA7);
        BN7.buildClassifier(banknoty);
        Evaluation eval_train7 = new Evaluation(banknoty);
        eval_train7.evaluateModel(BN7,banknoty);
        System.out.println("==========Sieæ Bayesowska z algorytmem szukania:LAGDHillClimber==========");
        System.out.println(eval_train7.toSummaryString());
        System.out.println(eval_train7.toMatrixString());
       
	        // perform cross-validation
	    for (int i = 0; i < runs; i++) {
	      // randomize data
	      int seed = i + 1;
	      java.util.Random rand = new java.util.Random(seed);
	      Instances randData = new Instances(banknoty);
	      randData.randomize(rand);
	      if (randData.classAttribute().isNominal())
	        randData.stratify(folds);
	 
	      Evaluation eval = new Evaluation(randData);
	      for (int n = 0; n < folds; n++) {
	        Instances train = randData.trainCV(folds, n);
	        Instances test = randData.testCV(folds, n);
	       
	       
	        BN7.buildClassifier(train);
	        eval.evaluateModel(BN7, test);
	      }
	 
	      // output evaluation
	      System.out.println();
	      System.out.println("=== Sieæ Bayesowska z algorytmem szukania:LAGDHillClimber run " + (i+1) + " ===");
	      System.out.println("Classifier: " + BN7.getClass().getName() + " " + Utils.joinOptions(BN7.getOptions()));
	      System.out.println("Dataset: " + banknoty.relationName());
	      System.out.println("Folds: " + folds);
	      System.out.println("Seed: " + seed);
	      System.out.println();
	      System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run " + (i+1) + "===", false)+eval.toMatrixString());
	       
	    }
       
        SearchAlgorithm SA8 = new RepeatedHillClimber();
        BayesNet BN8 = new BayesNet();
        BN8.setSearchAlgorithm(SA8);
        BN8.buildClassifier(banknoty);
        Evaluation eval_train8 = new Evaluation(banknoty);
        eval_train8.evaluateModel(BN8,banknoty);
        System.out.println("==========Sieæ Bayesowska z algorytmem szukania:RepeatedHillClimber==========");
        System.out.println(eval_train8.toSummaryString());
        System.out.println(eval_train8.toMatrixString());
       
        // perform cross-validation
	    for (int i = 0; i < runs; i++) {
	      // randomize data
	      int seed = i + 1;
	      java.util.Random rand = new java.util.Random(seed);
	      Instances randData = new Instances(banknoty);
	      randData.randomize(rand);
	      if (randData.classAttribute().isNominal())
	        randData.stratify(folds);
	 
	      Evaluation eval = new Evaluation(randData);
	      for (int n = 0; n < folds; n++) {
	        Instances train = randData.trainCV(folds, n);
	        Instances test = randData.testCV(folds, n);
	       
	       
	        BN8.buildClassifier(train);
	        eval.evaluateModel(BN8, test);
	      }
	 
	      // output evaluation
	      System.out.println();
	      System.out.println("=== Sieæ Bayesowska z algorytmem szukania:RepeatedHillClimber run " + (i+1) + " ===");
	      System.out.println("Classifier: " + BN8.getClass().getName() + " " + Utils.joinOptions(BN8.getOptions()));
	      System.out.println("Dataset: " + banknoty.relationName());
	      System.out.println("Folds: " + folds);
	      System.out.println("Seed: " + seed);
	      System.out.println();
	      System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation run " + (i+1) + "===", false)+eval.toMatrixString());
	       
	    }
        
        /*======================================================================
         *====================================================================== 
         * Istotnosc atrybutow
         *======================================================================
         *======================================================================
         */        
		
		ReliefFAttributeEval RFE = new ReliefFAttributeEval();
        RFE.buildEvaluator(banknoty);
        System.out.println(RFE);
        System.out.println("Istotnoœci poszczególnych atrybutów:");
        System.out.println(RFE.evaluateAttribute(0)+" - Wariancja");
        System.out.println(RFE.evaluateAttribute(1)+" - Skoœnoœæ");
        System.out.println(RFE.evaluateAttribute(2)+" - Kurtoza");
        System.out.println(RFE.evaluateAttribute(3)+" - Entropia");
  

	}

}