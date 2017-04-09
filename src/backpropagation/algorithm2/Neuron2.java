/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backpropagation.algorithm2;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 *
 * @author tom
 */
public class Neuron2 {
    
    private int inputsCount;
    private double learningRate;
    private double influenceRate;
    private List<Double> weights;
    private List<Double> prevWeightChanges;
    private List<Double> lastInputs;
    private double lastY;
    private boolean isOutput;
    private double delta;
    private int position;
    
    public final static double LAMBDA = 1;
    
    public Neuron2(int inputsCount, double learningRate, double influenceRate, 
            boolean inOutputLayer, int positionInLayer)
    {
        this.inputsCount = inputsCount;
        this.learningRate = learningRate;
        this.influenceRate = influenceRate;
        this.weights = new ArrayList<>();
        this.prevWeightChanges = new ArrayList<>();
        this.lastY = 0;
        this.isOutput = inOutputLayer;
        this.position = positionInLayer;
        Random gntr = new Random();
        for (int i = 0; i < inputsCount; i++)
        {
            double w = gntr.nextDouble() - 0.5;
            this.weights.add(w);
            this.prevWeightChanges.add(0.0);
        }
    }

    public double getLastY() {
        return lastY;
    }

    public double getDelta() {
        return delta;
    }
    
    public double getWeight(int index) {
        return this.weights.get(index);
    }
    
    public void markOutput() {
        this.isOutput = true;
    }
    
    public double funY(List<Double> inputs)
    {
        this.lastInputs = new ArrayList<>(inputs);
        if (inputs.size() != this.inputsCount)
        {
            throw new RuntimeException("Invalid size of input. Expected: " 
                    + inputsCount + ". Entered: " + inputs.size());
        }
        
        double z = 0;
        
        Iterator<Double> ws = this.weights.iterator();
        Iterator<Double> is = inputs.iterator();
        while (ws.hasNext() && is.hasNext())
        {
            z += ws.next() * is.next();
        }
        
        double y = 1.0 / ( 1.0 + Math.pow( Math.E, (-LAMBDA * z) ) );
        this.lastY = y;
        
        return y;
    }
    
    public void computeDelta(List<Neuron2> upperLayer, double expectedOutput)
    {
        if (isOutput)
        {
            this.delta = this.lastY - expectedOutput;
        }
        else
        {
            this.delta = 0.0;
            for (Neuron2 n : upperLayer)
            {
                this.delta = n.getDelta() * n.getLastY() * (1.0 - n.getLastY()) 
                        * LAMBDA * n.getWeight(this.position);
            }
        }
    }
    
    public void updateWeights()
    {
        List<Double> tmp_ws = new ArrayList<>();
        List<Double> tmp_pws = new ArrayList<>();
        Iterator<Double> ws = this.weights.iterator();
        Iterator<Double> xs = this.lastInputs.iterator();
        Iterator<Double> pws = this.prevWeightChanges.iterator();
        while (ws.hasNext() && xs.hasNext() && pws.hasNext())
        {
            double w = ws.next();
            double x = xs.next();
            double pwc = pws.next();
            double change = ( (-1.0) * this.learningRate * this.delta * 
                    this.lastY * (1.0 - this.lastY) * LAMBDA * x ) 
                    + this.influenceRate * pwc;
            double newW = w + change;
            tmp_ws.add(newW);
            tmp_pws.add(change);
        }
        
        this.prevWeightChanges = tmp_pws;
        this.weights = tmp_ws;
    }
    
}
