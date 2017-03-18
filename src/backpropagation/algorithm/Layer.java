/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backpropagation.algorithm;

import java.util.ArrayList;

/**
 *
 * @author tom
 */
public class Layer extends ArrayList<Neuron> {
    
    private double[] ys;
    private boolean isOutput = false;
    
    public void process(double[] inputs, double[] expectedOutputs)
    {
        this.ys = new double[this.size()];
        int i_outputs = 0;
        for (Neuron n : this)
        {
            n.computeY(inputs);
            if (isOutput) n.setExpectedOutput(expectedOutputs[i_outputs]);
            this.ys[i_outputs++] = n.getY();
        }
    }
    
    public double[] getYs()
    {
        return this.ys;
    }
    
    public void markAsOutput()
    {
        for (Neuron n: this)
        {
            n.markAsOutput();
        }
    }
    
    public void propagate(Layer prev)
    {
        int i = 0;
        for (Neuron n: this)
        {
            n.propagate(prev, i++);
        }
    }
            
    
}
