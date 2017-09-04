package eksploracja;

import weka.attributeSelection.ReliefFAttributeEval;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.core.EuclideanDistance;
import weka.core.FilteredDistance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.unsupervised.attribute.RandomProjection;
import static weka.filters.unsupervised.attribute.RandomProjection.SPARSE1;
import static weka.filters.unsupervised.attribute.RandomProjection.TAGS_DSTRS_TYPE;

//import java.util.Arrays;
//import java.io.*;
//import java.util.*;


public class Eksploracja {

	public static void main(String[] args) throws Exception {
		
		Instances banknoty = DataSource.read("./sampledata/banknote-authentication.arff");
		
		Instances banknoty1 = DataSource.read("./sampledata/banknote-authentication_part01.arff");
		Instances banknoty2 = DataSource.read("./sampledata/banknote-authentication_part02.arff");
        //System.out.println("Banknoty: "+banknoty);
        banknoty.setClassIndex(4);
        banknoty1.setClassIndex(4);
        banknoty2.setClassIndex(4);
        
        //Istotność atrybutów
        ReliefFAttributeEval RFE = new ReliefFAttributeEval();
        RFE.buildEvaluator(banknoty);
        System.out.println(RFE);
        System.out.println("Istotności poszczególnych atrybutów:");
        System.out.println(RFE.evaluateAttribute(0)+" - Wariancja");
        System.out.println(RFE.evaluateAttribute(1)+" - Skośność");
        System.out.println(RFE.evaluateAttribute(2)+" - Kurtoza");
        System.out.println(RFE.evaluateAttribute(3)+" - Entropia");
        
        /*======================================================================
         *====================================================================== 
         * MultiLayer Perceptron
         *======================================================================
         *======================================================================
         */
        
        MultilayerPerceptron per = new MultilayerPerceptron();
        per.setTrainingTime(800);
        per.setMomentum(0.2);
        per.setLearningRate(0.3);
        per.setHiddenLayers("4");
        per.setValidationSetSize(10);
        per.setGUI(true);
        for (int i=0; i<per.getOptions().length; i++) {
        	System.out.print(per.getOptions()[i] + " ");
        }
        
        per.buildClassifier(banknoty);
        Evaluation ev = new Evaluation(banknoty);
        ev.evaluateModel(per, banknoty);
        
        
        System.out.println("\n=====================================================");
        System.out.println("MultiLayer Perceptron");
        System.out.println("=====================================================");
        System.out.println(ev.toSummaryString());
        System.out.println(ev.toClassDetailsString());
        System.out.println(ev.toMatrixString());
        
        
        /*======================================================================
         *====================================================================== 
         * IBk
         *======================================================================
         *======================================================================
         */
        
        
        //Tworzymy algorytm
        EuclideanDistance ed = new EuclideanDistance();
        ed.setAttributeIndices("first-last");
        
        RandomProjection rp = new RandomProjection();
        rp.setDistribution(new SelectedTag(SPARSE1, TAGS_DSTRS_TYPE));
        rp.setNumberOfAttributes(10);
        rp.setSeed(42);
       
        FilteredDistance fd = new FilteredDistance();
        fd.setAttributeIndices("first-last");
        fd.setDistance(ed);
        fd.setFilter(rp);
       
        LinearNNSearch lNNS = new LinearNNSearch();
        lNNS.setDistanceFunction(fd);
        lNNS.setMeasurePerformance(false);
        lNNS.setSkipIdentical(false);
       
        //PARAMETRY KLASYFIKATORA
        IBk ibk = new IBk();
       
        ibk.setKNN(7);
        ibk.setBatchSize("100");
        ibk.setCrossValidate(false);
        ibk.setDebug(false);
        //ibk.setDistanceWeighting(newMethod)
        ibk.setDoNotCheckCapabilities(false);
        ibk.setMeanSquared(false);
        ibk.setNearestNeighbourSearchAlgorithm(lNNS);
        ibk.setNumDecimalPlaces(10);
        ibk.setWindowSize(0);
       
       
        //Budowanie klasyfikatora
        ibk.buildClassifier(banknoty);
       
 
       
        //WYSWIETLENIE WSTEPNYCH PARAMETROW
        System.out.println("\n=====================================================");
        System.out.println("IBk");
        System.out.println("=====================================================");
        for (int i=0; i<ibk.getOptions().length; i++) {
            System.out.print(ibk.getOptions()[i] + " ");
        }
        System.out.println("\nParametry LinearNNSearch");
        for (int i=0; i<lNNS.getOptions().length; i++) {
            System.out.print(lNNS.getOptions()[i] + " ");
        }
        System.out.println("\nParametry Euclidean Distance");
        for (int i=0; i<ed.getOptions().length; i++) {
            System.out.print(ed.getOptions()[i] + " ");
        }
        System.out.println("\n=====================================================");
       
        //OGOLNIE WYNIKI
        Evaluation ev2 = new Evaluation(banknoty);
        ev2.evaluateModel(ibk, banknoty);
       
       
        //System.out.println("\n" + ev2.toCumulativeMarginDistributionString());
        System.out.println("\n" + ev2.toSummaryString());
        System.out.println("\n" + ev2.toClassDetailsString());
        System.out.println("\n" + ev2.toMatrixString());
   

	}

}