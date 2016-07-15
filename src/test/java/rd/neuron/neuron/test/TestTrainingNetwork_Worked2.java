/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rd.neuron.neuron.test;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.DataWriter;
import rd.data.FileDataWriter;
import rd.neuron.neuron.DataStreamer;
import rd.neuron.neuron.FullyRandomLayerBuilder;
import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.NetworkError;
import rd.neuron.neuron.SimpleNetwork;
import rd.neuron.neuron.SimpleNetworkPerformanceEvaluator;

/**
 *
 * @author azahar
 */
public class TestTrainingNetwork_Worked2 {

	private boolean SGD = true;
	private int EPOCHS = 100000;

	@Test
	public void doLayer() throws IOException {

		NetworkError e = new NetworkError();
		DataStreamer input = new DataStreamer(2, 1);
		input.add(new float[] { 1f, 1.0f }, 0f);
		input.add(new float[] { 0f, 0.0f }, 0f);
		input.add(new float[] { 1f, 0.0f }, 1f);

		input.add(new float[] { 0f, 1.0f }, 1f);

		SimpleNetwork network = new SimpleNetwork(new FullyRandomLayerBuilder(0.5f,1f), Function.LOGISTIC, 2, 2, 1);
		DataWriter dw = new FileDataWriter("weights_short.csv",true);
		SimpleNetworkPerformanceEvaluator snpe = new SimpleNetworkPerformanceEvaluator(dw);
		System.out.println(network.getNumberOfLayers());

		System.out.println("Number of layers: " + network.getNumberOfLayers() + "\n" + network);
		FloatMatrix first = input.iterator().next();
		// Back Prop
		for (int i = 0; i < EPOCHS; i++) {
			if (!SGD) {
				for (FloatMatrix item : input) {

					FloatMatrix outputLayerNewWts[] = network.trainOutputLayer(0.05f, input.getOutput(item),
							network.io(item));
					FloatMatrix hiddenLayerNewWts[] = network.trainHiddenLayer(0, 0.05f, input.getOutput(item),
							network.io(item), item);

					network.setOutputWeights(outputLayerNewWts[0]);
					network.setOutputBias(outputLayerNewWts[1]);
					network.setWeights(0, hiddenLayerNewWts[0]);
					network.setBias(0, hiddenLayerNewWts[1]);
				}
			} else {
				FloatMatrix item = input.getRandom();

				FloatMatrix outputLayerNewWts[] = network.trainOutputLayer(0.05f, input.getOutput(item),
						network.io(item));
				FloatMatrix hiddenLayerNewWts[] = network.trainHiddenLayer(0, 0.05f, input.getOutput(item),
						network.io(item), item);

				network.setOutputWeights(outputLayerNewWts[0]);
				network.setOutputBias(outputLayerNewWts[1]);
				network.setWeights(0, hiddenLayerNewWts[0]);
				network.setBias(0, hiddenLayerNewWts[1]);

			}
			snpe.evaluate(network, input);
		}
		for (FloatMatrix item : input) {
			System.out.println(item + " > Actual: " + network.io(item) + "  > Expected: " + input.getOutput(item));
		}
		System.out.println(network);
		
		dw.close();
	}
}
