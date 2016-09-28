/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rd.neuron.neuron.test;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.DataStreamer;
import rd.data.DataWriter;
import rd.data.FileDataWriter;
import rd.neuron.neuron.FullyRandomLayerBuilder;
import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.SimpleNetwork;
import rd.neuron.neuron.SimpleNetworkPerformanceEvaluator;
import rd.neuron.neuron.TrainNetwork;

/**
 * Network XOR implementation
 * 
 * @author azahar
 */
public class TestTrainingNetworkXORWithL2 {

	private int EPOCHS = 200000;
	private final float LEARNING_RATE = 0.05f;
	private final float BETA = 1f;

	@Test
	public void doLayer() throws IOException {

		DataStreamer input = new DataStreamer(2, 1);
		// XOR data set
		input.add(new float[] { 1f, 1.0f }, 0f);
		input.add(new float[] { 0f, 0.0f }, 0f);
		input.add(new float[] { 1f, 0.0f }, 1f);
		input.add(new float[] { 0f, 1.0f }, 1f);

		SimpleNetwork network = new SimpleNetwork(new FullyRandomLayerBuilder(0f, 1f), Function.LOGISTIC, 2, 2, 1);
		DataWriter dw = new FileDataWriter("weights_short.csv", true);
		SimpleNetworkPerformanceEvaluator snpe = new SimpleNetworkPerformanceEvaluator(dw);
		System.out.println(network.getNumberOfLayers());

		System.out.println("Number of layers: " + network.getNumberOfLayers() + "\n" + network);

		// Back Prop
		for (int i = 0; i < EPOCHS; i++) {
			FloatMatrix item = input.getRandom();
			TrainNetwork.trainBackpropWithL2(network, item, input.getOutput(item), LEARNING_RATE,BETA);
			snpe.evaluateErrorAndNetwork(network, input, LEARNING_RATE);
		}
		
		Map<Integer,FloatMatrix> activation = new HashMap<>();
		for (FloatMatrix item : input) {
			System.out.println(item + " > Actual: " + network.io(item,activation) + "  > Expected: " + input.getOutput(item)+"\n"+activation);
			
		}
		System.out.println(network);

		dw.close();
	}
}
