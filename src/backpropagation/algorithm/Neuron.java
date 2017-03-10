/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backpropagation.algorithm;

import java.util.Random;

/**
 *
 * @author tom
 */
public class Neuron {
    
    private double[] weights;
    private double output;
    
    public static double LAMBDA = 5000;
    
    public Neuron(int numOfInputs)
    {
        this.weights = new double[numOfInputs];
        Random generator = new Random();
        for (int i = 0; i < numOfInputs; i++)
        {
            this.weights[i] = generator.nextDouble();
        }
    }
    
    public double getWeight(int index)
    {
        if (index >= this.weights.length)
            throw new RuntimeException("Set weight index out of bound!");
        return this.weights[index];
    }
    
    public double process(double[] inputs)
    {
        double z = 0;
        for (int i = 0; i < this.weights.length; i++)
        {
            z += inputs[i] * this.weights[i];
        }
        
        this.output = 1 / ( 1 + Math.pow(Math.E, (-Neuron.LAMBDA * z) ) );
        
        return this.output;
    }

    public double getOutput() {
        return output;
    }
    
}
