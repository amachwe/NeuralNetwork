/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rd.neuron.neuron.test;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.neuron.neuron.Layer.Function;
import rd.data.DataStreamer;
import rd.neuron.neuron.NetworkError;
import rd.neuron.neuron.SimpleNetwork;
import rd.neuron.neuron.UnitLayerBuilder;

/**
 *
 * @author azahar
 */
public class TestInputNetwork {

	@Test
	public void doLayer() {

		NetworkError e = new NetworkError();
		DataStreamer input = new DataStreamer(3, 1);
		input.add(new float[] { 1, 1, 1 }, 0f);
		input.add(new float[] { 0, 1, 1 }, 1f);
		input.add(new float[] { 0, 1, 0 }, 0f);

		SimpleNetwork network = new SimpleNetwork(new UnitLayerBuilder(), Function.ReLU, 3, 5, 1);
		System.out.println("Number of layers: "+network.getNumberOfLayers()+"\n"+network);
		//Stochastic
		for (int i = 0; i < 100; i++) {
			e.reset();
			for (FloatMatrix item : input) {
				e.localError(network.io(item), input.getOutput(item));
			}
			System.out.println("Old: " + e.getError());
			float prevError = e.getError();

			float w1 = network.adjustWeight(0, 0, 0, network.getWeight(0, 0, 0) + (float) (0.5 - Math.random()));
			float w2 = network.adjustWeight(1, 0, 0, network.getWeight(1, 0, 0) + (float) (0.5 - Math.random()));
			float w3 = network.adjustWeight(0, 1, 1, network.getWeight(0, 1, 0) + (float) (0.5 - Math.random()));
			float w4 = network.adjustWeight(1, 0, 1, network.getWeight(1, 0, 1) + (float) (0.5 - Math.random()));

			e.reset();
			for (FloatMatrix item : input) {
				e.localError(network.io(item), input.getOutput(item));

			}

			System.out.println("New: " + e.getError() + "   > " + (prevError - e.getError()));
			if (prevError - e.getError() <= 0) {
				network.adjustWeight(0, 0, 0, w1);
				network.adjustWeight(1, 0, 0, w2);
				network.adjustWeight(0, 1, 1, w3);
				network.adjustWeight(1, 0, 1, w4);
			}

		}
		for (FloatMatrix item : input) {
			System.out.println(item+" > Actual: "+network.io(item)+ "  > Expected: "+input.getOutput(item));
		}
	}
}
