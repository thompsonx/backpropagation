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

/**
 *
 * @author tom
 */
public class Network {

    private BackpropagationNeuronNet netdata;
    private List<Layer> network;

    public Network(BackpropagationNeuronNet netdata) {
        this.netdata = netdata;
        
        int[] inputsPerLayer = new int[netdata.getLayersCount()];
        inputsPerLayer[0] = netdata.getInputsCount();
        for (int i = 1; i < netdata.getLayersCount() - 1; i++)
            inputsPerLayer[i] = netdata.getNeuronInLayersCount()[i - 1];
        
        this.network = new ArrayList<>(netdata.getLayersCount());
        int i_input = 0;
        for (Layer l : this.network)
        {
            for (int i = 0; i < netdata.getNeuronInLayersCount()[i_input]; i++)
                l.add(new Neuron(inputsPerLayer[i_input]));
            i_input++;
        }
    }
    
    public void doBackPropagation()
    {
        // Forward propagation
        for (TrainSetElement tse : this.netdata.getTrainingSet())
        {
            double[] inputs = tse.getInputs();
            for (Layer l : this.network)
            {
                l.process(inputs);
                inputs = l.getOutputs();
            }
        }
    }
    
    
}
