/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backpropagation.algorithm2;

import backpropagation.data.BackpropagationNeuronNet;
import backpropagation.data.TrainSetElement;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 *
 * @author tom
 */
public class Network2 {
    
    private BackpropagationNeuronNet netdata;
    private List<List<Neuron2>> net;
    private double netError;

    public Network2(BackpropagationNeuronNet netdata) {
        this.netdata = netdata;
        this.buildNet();
    }

    public double getNetError() {
        return netError;
    }
    
    private void buildNet()
    {
        int layers = this.netdata.getLayersCount();
        int inputs = this.netdata.getInputsCount();
        this.net = new ArrayList<>();
        for (int i = 0; i < layers; i++)
        {
            int neurons = this.netdata.getNeuronInLayersCount()[i];
            List<Neuron2> layer = new ArrayList<>();
            for (int n = 0; n < neurons; n++)
            {
                layer.add(new Neuron2(inputs,
                                      netdata.getLearningRate(), 
                                      netdata.getLastStepInfluenceLearningRate(),
                                      false,
                                      n));
            }
            this.net.add(layer);
            inputs = neurons;
        }
        
        for (Neuron2 n : this.net.get(this.net.size() - 1))
        {
            n.markOutput();
        }
    }
    
    public void learn()
    {
        List<TrainSetElement> trainSet = this.netdata.getTrainingSet();
        Collections.shuffle(trainSet);
        for (TrainSetElement element : trainSet)
        {
            List<Double> inputs = new ArrayList<>();
            for (double i : element.getInputs())
            {
                inputs.add(i);
            }
            for (List<Neuron2> layer : this.net)
            {
                List<Double> new_inputs = new ArrayList<>();
                for (Neuron2 neuron : layer)
                {
                    double out = neuron.funY(inputs);
                    new_inputs.add(out);
                }
                inputs = new_inputs;                
            }
            
            List<Neuron2> outLayer = this.net.get(this.net.size() - 1);
            int outputN = 0;
            for (Neuron2 neuron : outLayer)
            {
                neuron.computeDelta(null, element.getOutput()[outputN++]);
            }
            List<Neuron2> upperLayer = outLayer;
            for (int l = this.net.size() - 2; l >= 0; l--)
            {
                List<Neuron2> layer = this.net.get(l);
                for (Neuron2 neuron : layer)
                {
                    neuron.computeDelta(upperLayer, 0);
                }
                upperLayer = layer;
            }
            
            for (List<Neuron2> layer : this.net)
            {
                for (Neuron2 neuron : layer)
                {
                    neuron.updateWeights();
                }
            }
        }
        
        double error = 0;
        for (TrainSetElement element : trainSet)
        {
            List<Double> inputs = new ArrayList<>();
            for (double i : element.getInputs())
            {
                inputs.add(i);
            }
            List<Double> output = this.evaluate(inputs);
            int outputN = 0;
            for (double o : output)
            {
                error += Math.pow( ( o - element.getOutput()[outputN++] ) , 2);
            }
        }
        
        error *= 0.5;
        this.netError = error;
    }
    
    public List<Double> evaluate(List<Double> netInputs)
    {
        List<Double> inputs = netInputs;
        for (List<Neuron2> layer : this.net)
        {
            List<Double> new_inputs = new ArrayList<>();
            for (Neuron2 neuron : layer)
            {
                new_inputs.add(neuron.funY(inputs));
            }
            inputs = new_inputs;                
        }
        
        return inputs;
    }
}
