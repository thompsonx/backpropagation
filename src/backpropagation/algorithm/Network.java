/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backpropagation.algorithm;

import backpropagation.data.BackpropagationNeuronNet;
import backpropagation.data.TrainSetElement;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

/**
 *
 * @author tom
 */
public class Network {

    private BackpropagationNeuronNet netdata;
    private List<Layer> network;
    private double error;

    public Network(BackpropagationNeuronNet netdata) {
        this.netdata = netdata;
        
        int[] inputsPerLayer = new int[netdata.getLayersCount()];
        inputsPerLayer[0] = netdata.getInputsCount();
        for (int i = 1; i < netdata.getLayersCount(); i++)
            inputsPerLayer[i] = netdata.getNeuronInLayersCount()[i - 1];
        
        this.network = new ArrayList<>();
        for (int i_input = 0; i_input < netdata.getLayersCount(); i_input++)
        {
            Layer l = new Layer();
            for (int i = 0; i < netdata.getNeuronInLayersCount()[i_input]; i++)
                l.add(new Neuron(inputsPerLayer[i_input], 
                        netdata.getLearningRate(),
                        netdata.getLastStepInfluenceLearningRate()));
            this.network.add(l);
        }
        this.network.get( this.network.size() - 1 ).markAsOutput();
    }

    public double getError() {
        return error;
    }
    
    public void learn()
    {
        double prev_error = Double.MAX_VALUE;
        double cur_error = Double.MAX_VALUE;
        
        while (cur_error <= prev_error)
        {
            prev_error = cur_error;
            for (TrainSetElement tse : this.netdata.getTrainingSet())
            {
                // Forward propagation
                double[] inputs = tse.getInputs();
                for (Layer l : this.network)
                {
                    l.learn(inputs, tse.getOutput());
                    inputs = l.getYs();
                }

                // Backward propagation
                ListIterator<Layer> iter = this.network.listIterator(network.size());
                Layer previous = null;
                while (iter.hasPrevious())
                {
                    Layer l = iter.previous();
                    l.propagate(previous);
                    previous = l;
                }
            }

            // Global error determination
            cur_error = 0;
            for (TrainSetElement tse: this.netdata.getTrainingSet())
            {
                // Forward propagation
                double[] inputs = tse.getInputs();
                for (Layer l : this.network)
                {
                    l.learn(inputs, tse.getOutput());
                    inputs = l.getYs();
                }
                double[] outputs = inputs;
                int index = 0;
                // All neurons in output layer
                for (double o : outputs)
                {
                    cur_error += Math.pow(o - tse.getOutput()[index++], 2);
                }
            }
        
            cur_error *= 0.5;
        }
        
        this.error = cur_error;
    }
    
    public double[] evaluate(double[] inputs)
    {
        double[] l_inputs = inputs;
        for (Layer l : this.network)
        {
            l.evaluate(l_inputs);
            l_inputs = l.getYs();
        }
        
        return l_inputs;
    }
    
}
