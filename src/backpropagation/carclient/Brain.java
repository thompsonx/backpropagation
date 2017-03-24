/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backpropagation.carclient;

import backpropagation.algorithm.Network;
import backpropagation.data.BackpropagationNeuronNet;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.HashMap;
import javax.xml.bind.JAXBException;

/**
 *
 * @author tom
 */
public class Brain implements DriverInterface {

    private Network net;
    
    public Brain() throws JAXBException, FileNotFoundException
    {
        FileReader fr = new FileReader("brain.xml");
        this.net = new Network(BackpropagationNeuronNet.readFromXml(fr));
        this.net.learn();
    }
    
    @Override
    public HashMap<String, Float> drive(HashMap<String, Float> values) {
        double[] vals = new double[values.values().size()];
        int i = 0;
        for (Float f : values.values())
        {
            vals[i++] = f;
        }
        double[] output = this.net.evaluate(vals);
        HashMap<String, Float> response = new HashMap<>();
        response.put("wheel", (float)output[0]);
        response.put("acc", (float)output[1]);
        
        return response;
    }
    
}
