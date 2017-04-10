/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backpropagation.carclient;

import backpropagation.algorithm2.Network2;
import backpropagation.data.BackpropagationNeuronNet;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import javax.xml.bind.JAXBException;

/**
 *
 * @author tom
 */
public class Brain implements DriverInterface {

    private Network2 net;
    public final static String[] KEYS = {
        "angle","speed","distance0","distance4","distance8","distance16",
        "distance32","friction","skid","checkpoint","sensorFrontLeft",
        "sensorFrontMiddleLeft","sensorFrontMiddleRight","sensorFrontRight",
        "sensorFrontRightCorner1","sensorFrontRightCorner2","sensorRight1",
        "sensorRight2","sensorRearRightCorner2","sensorRearRightCorner1",
        "sensorRearRight","sensorRearLeft","sensorRearLeftCorner1",
        "sensorRearLeftCorner2","sensorLeft1","sensorLeft2",
        "sensorFrontLeftCorner1","sensorFrontLeftCorner2"
    };
    
    public Brain() throws JAXBException, FileNotFoundException
    {
        FileReader fr = new FileReader("brain.xml");
        BackpropagationNeuronNet data = BackpropagationNeuronNet.readFromXml(fr);       
        this.net = new Network2(data);
        for (int i = 0; i < 1000; i++)
        {
            this.net.learn();
        }
        
        System.out.println("Error: " + net.getNetError());
    }
    
    @Override
    public HashMap<String, Float> drive(HashMap<String, Float> values) {
        List<Double> vals = new ArrayList<>();
        for (String key : KEYS)
        {
            vals.add((double)values.get(key));
        }
        List<Double> output = this.net.evaluate(vals);
        HashMap<String, Float> response = new HashMap<>();
        double wheel = output.get(0);
        double acc = output.get(1);
        response.put("wheel", (float)wheel);
        response.put("acc", (float)acc);
        return response;
    }
    
}
