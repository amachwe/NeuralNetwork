/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rd.neuron.neuron.test;

import static org.junit.Assert.assertTrue;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.DataStreamer;
import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.SimpleNetwork;
import rd.neuron.neuron.TrainNetwork;
import rd.neuron.neuron.UnitLayerBuilder;

/**
 * To validate the working of the network using values derived from:
 * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 * 
 * @author azahar
 */
public class TestTrainingNetworkExample {

	@Test
	public void doLayer() {

		DataStreamer input = new DataStreamer(2, 2);
		input.add(new float[] { 0.05f, 0.10f }, 0.01f, 0.99f);

		SimpleNetwork network = new SimpleNetwork(new UnitLayerBuilder(), Function.LOGISTIC, 2, 2, 2);
		System.out.println(network.getNumberOfLayers());
		// Prepare the network with weights and biases as given in the worked
		// example
		network.adjustWeight(0, 0, 0, 0.15f);
		network.adjustWeight(0, 0, 1, 0.20f);
		network.adjustWeight(0, 1, 0, 0.25f);
		network.adjustWeight(0, 1, 1, 0.30f);

		network.adjustWeight(1, 0, 0, 0.40f);
		network.adjustWeight(1, 0, 1, 0.45f);
		network.adjustWeight(1, 1, 0, 0.50f);
		network.adjustWeight(1, 1, 1, 0.55f);

		network.adjustBias(0, 0, 0.35f);
		network.adjustBias(0, 1, 0.35f);

		network.adjustBias(1, 0, 0.6f);
		network.adjustBias(1, 1, 0.6f);

		System.out.println("Number of layers: " + network.getNumberOfLayers() + "\n" + network);

		// Back Prop
		for (int i = 0; i < 10000; i++) {
			for (FloatMatrix item : input) {

				TrainNetwork.train(network, item, input.getOutput(item), 0.5f);

				if (i == 0) {
					assertTrue(network.getWeight(1, 0, 0) == 0.3589165f);
					assertTrue(network.getWeight(1, 1, 1) == 0.56137013f);
					assertTrue(network.getWeight(1, 0, 1) == 0.40866616f);
					assertTrue(network.getWeight(1, 1, 0) == 0.5113013f);

					assertTrue(network.getWeight(0, 0, 0) == 0.14978072f);
					assertTrue(network.getWeight(0, 1, 1) == 0.2995023f);
					assertTrue(network.getWeight(0, 1, 0) == 0.24975115f);
					assertTrue(network.getWeight(0, 0, 1) == 0.19956143f);
				}

			}

		}
		for (FloatMatrix item : input) {
			System.out.println(item + " > Actual: " + network.io(item) + "  > Expected: " + input.getOutput(item));
		}

		System.out.println(input.iterator().next() + " > Actual: " + network.io(input.iterator().next())
				+ "  > Expected: " + input.getOutput(input.iterator().next()));

	}
}
