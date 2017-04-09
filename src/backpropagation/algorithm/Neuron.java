/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backpropagation.algorithm;

import java.util.Arrays;
import java.util.Random;

/**
 *
 * @author tom
 */
public class Neuron {
    
    private double[] weights;
    private double[] prevWeights;
    private double[] prevWeightChanges;
    private double lastY;
    private double lastDelta;
    private double[] lastInputs;
    private boolean isOutput;
    private double expectedOutput;
    private double lrate;
    private double influenceRate;
    
    public static double LAMBDA = 0.7;
    
    public Neuron(int numOfInputs, double lrate, double influenceRate)
    {
        this.isOutput = false;
        this.lrate = lrate;
        this.influenceRate = influenceRate;
        this.weights = new double[numOfInputs];
        this.prevWeightChanges = new double[numOfInputs];
        this.prevWeights = new double[numOfInputs];
        this.lastInputs = new double[numOfInputs];
        Random generator = new Random();
        for (int i = 0; i < numOfInputs; i++)
        {
            this.weights[i] = this.prevWeights[i] = generator.nextDouble();
            this.prevWeightChanges[i] = 0;
            this.lastInputs[i] = 0;
        }
    }
    
    public Neuron(Neuron n)
    {
        this.isOutput = n.isOutput;
        this.lrate = n.lrate;
        this.influenceRate = n.influenceRate;
        this.weights = new double[n.weights.length];
        this.prevWeightChanges = new double[n.weights.length];
        this.prevWeights = new double[n.weights.length];
        this.lastInputs = new double[n.weights.length];
        this.expectedOutput = n.expectedOutput;
        this.lastY = n.lastY;
        this.lastDelta = n.lastDelta;
        for (int i = 0; i < n.weights.length; i++)
        {
            this.weights[i] = n.weights[i];
            this.prevWeights[i] = n.prevWeights[i];
            this.prevWeightChanges[i] = n.prevWeightChanges[i];
            this.lastInputs[i] = n.lastInputs[i];
        }
    }

    public void setExpectedOutput(double expectedOutput) {
        this.expectedOutput = expectedOutput;
    }

    public double getDelta() {
        return lastDelta;
    }
    
    public void markAsOutput()
    {
        this.isOutput = true;
    }
    
    public double getWeight(int index)
    {
        if (index >= this.weights.length)
            throw new RuntimeException("Set weight index out of bound!");
        return this.weights[index];
    }
    
    public double getPrevWeight(int index)
    {
        if (index >= this.prevWeights.length)
            throw new RuntimeException("Set weight index out of bound!");
        return this.prevWeights[index];
    }
    
    public double computeY(double[] inputs)
    {
        this.lastInputs = inputs;
        double z = 0;
        for (int i = 0; i < this.weights.length; i++)
        {
            z += inputs[i] * this.weights[i];
        }
        
        this.lastY = 1.0 / ( 1.0 + Math.pow(Math.E, (-Neuron.LAMBDA * z) ) );
        
        return this.lastY;
    }

    public double getY() {
        return this.lastY;
    }
    
    public void propagate(Layer prev, int position)
    {
        double common = -lrate * LAMBDA * lastY * (1.0 - lastY);
        
        if (isOutput)
        {
            this.lastDelta = lastY - expectedOutput;
        }
        else 
        {
            this.lastDelta = 0;
            for (Neuron n: prev)
            {
                lastDelta += n.getDelta() * n.getY() * (1.0 - n.getY()) * LAMBDA *
                        n.getPrevWeight(position);
            }
        }
        
        for (int i = 0; i < this.weights.length; i++)
        {
            double change = common * lastInputs[i] * lastDelta
                    + influenceRate * prevWeightChanges[i];
            this.prevWeights[i] = weights[i];
            weights[i] += change;
            prevWeightChanges[i] = change;
        }
        
        System.out.println(lrate);
        
    }
    
}
