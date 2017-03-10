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
    
    private double[] outputs;
    
    public void process(double[] inputs)
    {
        this.outputs = new double[this.size()];
        int i_outputs = 0;
        for (Neuron n : this)
        {
            n.process(inputs);
            this.outputs[i_outputs++] = n.getOutput();
        }
    }
    
    public double[] getOutputs()
    {
        return this.outputs;
    }
    
}
