/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backpropagation;

import backpropagation.algorithm.Network;
import backpropagation.data.BackpropagationNeuronNet;
import java.io.FileNotFoundException;
import java.io.FileReader;
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
        Network net = new Network(BackpropagationNeuronNet.readFromXml(fr));
        net.learn();
    }
    
}
