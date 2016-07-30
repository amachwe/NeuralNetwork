/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rd.neuron.neuron.test;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.DataStreamer;
import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.NetworkError;
import rd.neuron.neuron.SimpleNetwork;
import rd.neuron.neuron.TrainNetwork;
import rd.neuron.neuron.UnitLayerBuilder;

/**
 *
 * @author azahar
 */
public class TestTrainingNetwork {

	@Test
	public void doLayer() {

		NetworkError e = new NetworkError();
		DataStreamer input = new DataStreamer(2, 1);
		input.add(new float[] { 1, 1 }, 1f);
		input.add(new float[] { 0, 1 }, 0f);
		input.add(new float[] { 1, 0 }, 0f);
		
		DataStreamer test = new DataStreamer(2,1);
		test.add(new float[] {0, 0},1f);

		SimpleNetwork network = new SimpleNetwork(new UnitLayerBuilder(), Function.LOGISTIC, 2, 2, 1);
	
		System.out.println("Number of layers: "+network.getNumberOfLayers()+"\n"+network);
		FloatMatrix first = input.iterator().next();
		// Stochastic
		for (int i = 0; i < 100000; i++) {
			FloatMatrix item = input.getRandom();
			TrainNetwork.train(network,item,input.getOutput(item),0.05f);

		}
		for (FloatMatrix item : input) {
			System.out.println(item + " > Actual: " + network.io(item) + "  > Expected: " + input.getOutput(item));
		}
		
		System.out.println(test.iterator().next() + " > Actual: " + network.io(test.iterator().next()) + "  > Expected: " + test.getOutput(test.iterator().next()));
		
	}
}
