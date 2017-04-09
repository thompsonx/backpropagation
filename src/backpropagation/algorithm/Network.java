/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backpropagation.algorithm;

import backpropagation.data.BackpropagationNeuronNet;
import backpropagation.data.TrainSetElement;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;
import java.util.Random;

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
    
    public Network(Network n)
    {
        this.netdata = n.netdata;
        this.network = new ArrayList<>();
        for (Layer l : n.network)
        {
            this.network.add(new Layer(l));
        }
        this.error = n.error;
    }

    public double getError() {
        return error;
    }
    
    public void learn()
    {
        ArrayList<TrainSetElement> trainSet = this.netdata.getTrainingSet();
        Collections.shuffle(trainSet, new Random());
        for (TrainSetElement tse : trainSet)
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
        double err = 0;
        for (TrainSetElement tse: trainSet)
        {
            double[] inputs = tse.getInputs();
            double[] outputs = this.evaluate(inputs);
            int index = 0;
            // All neurons in output layer
            for (double o : outputs)
            {
                err += Math.pow(o - tse.getOutput()[index++], 2);
            }
        }

        err *= 0.5;
        this.error = err;
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
