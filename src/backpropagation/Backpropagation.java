/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backpropagation;

import backpropagation.algorithm2.Network2;
import backpropagation.data.BackpropagationNeuronNet;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javax.xml.bind.JAXBException;

/**
 *
 * @author tom
 */
public class Backpropagation {

    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     * @throws javax.xml.bind.JAXBException
     */
    public static void main(String[] args) throws FileNotFoundException, JAXBException {
        if (args.length != 1)
            return;
        FileReader fr = new FileReader(args[0]);
        BackpropagationNeuronNet data = BackpropagationNeuronNet.readFromXml(fr);
        
        Network2 net = new Network2(data);
        net.learn();
        for (int e = 0; e < 10000; e++)
        {
            net.learn();
        }
        
        System.out.println("Error: " + Double.toString(net.getNetError()));
        
//  LEKAR
//        List<Double> input = new ArrayList<>();
//        input.add(42.0);
//        input.add(100.0);
//        input.add(50.0);
//        input.add(50.0);
//        input.add(50.0);
//        List<Double> output = net.evaluate(input);
//        for (double d : output)
//        {
//            System.out.println(d);
//        }
        
        List<Double> input = new ArrayList<>();
        input.add(0.0); // rychlost
        input.add(1.0); // spotreba
        input.add(0.0); // prostor
        input.add(0.5); // bezpecnost

        List<Double> output = net.evaluate(input);
        int i = 1;
        int max_i = 1;
        double max = -1;
        for (double d : output)
        {
            System.out.println(d);
            if (d > max)
            {
                max = d;
                max_i = i;
            }
            i++;
        }
        System.out.println("CHOICE: " + max_i + " (" + max + ")");
        System.out.println("1 - SUV, 2 - FAMILY, 3 - SPORT");
    }
    
}
