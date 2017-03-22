/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backpropagation.carclient;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 *
 * @author tom
 */
public class Client {
    
    
    public static void main(String[] args) throws IOException
    {
        String host = "localhost";
        int port = 9461;
        String raceName = "test";
        String driverName = "basic_client";
        String carType = null;
        RaceConnector raceConnector = null;
        if (args.length < 4) {
                // kontrola argumentu programu
                raceConnector = new RaceConnector(host, port, null);
                System.err.println("argumenty: server port nazev_zavodu jmeno_ridice [typ_auta]");
                List<String> raceList =  raceConnector.listRaces();
                raceName = raceList.get(new Random().nextInt(raceList.size()));
                List<String> carList =  raceConnector.listCars(raceName);
                carType = carList.get(new Random().nextInt(carList.size()));
                driverName += "_" + carType;
        } else {
                // nacteni parametu
                host = args[0];
                port = Integer.parseInt(args[1]);
                raceName = args[2];
                driverName = args[3];
                if(args.length > 4){
                        carType = args[4];
                }
                raceConnector = new RaceConnector(host, port, null);
        }
        // vytvoreni klienta
        //raceConnector.setDriver();
        raceConnector.start(raceName, driverName, carType);
    }
}
