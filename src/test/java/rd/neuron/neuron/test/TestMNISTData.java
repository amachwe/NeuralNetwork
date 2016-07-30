package rd.neuron.neuron.test;

import java.io.IOException;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.DataStreamer;
import rd.data.DataWriter;
import rd.data.DeltaInspector;
import rd.data.MnistToDataStreamer;
import rd.data.MongoDeltaWriter;
import rd.neuron.neuron.FullyRandomLayerBuilder;
import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.SimpleNetwork;
import rd.neuron.neuron.TrainNetwork;

public class TestMNISTData {

	private final int EPOCHS = 1000000;
	private final int REPEATS = 1;
	private final boolean useSGD = true;
	private final float LEARNING_RATE = 0.05f;

	@Test
	public void doMNISTTest() throws IOException {
		DataStreamer streamerTrain = MnistToDataStreamer.createStreamer("d:\\ml stats\\mnist\\t10k-images.idx3-ubyte",
				"d:\\ml stats\\mnist\\t10k-labels.idx1-ubyte");
		System.out.println("Training Streamer Ready!");

		DataStreamer streamerTest = MnistToDataStreamer.createStreamer("d:\\ml stats\\mnist\\train-images.idx3-ubyte",
				"d:\\ml stats\\mnist\\train-labels.idx1-ubyte");
		System.out.println("Testing Streamer Ready!");
		
		DataWriter dw = new MongoDeltaWriter("localhost",27017,"neural","sgd_backprop");
		

		for (int round = 0; round < REPEATS; round++) {
			SimpleNetwork network = new SimpleNetwork(new FullyRandomLayerBuilder(0.5f, 1f), Function.LOGISTIC, 784, 15,
					10);
			network.activateDeepNetworkInspection(new DeltaInspector(1e-6,dw));

			// Back Prop
			for (int i = 0; i < EPOCHS; i++) {
				if ((i * 100f / EPOCHS) % 10 == 0) {
					System.out.println(i * 100f / EPOCHS);

				}
				if (useSGD) {
					FloatMatrix input = streamerTrain.getRandom();
					TrainNetwork.train(network, input, streamerTrain.getOutput(input), LEARNING_RATE);
			
					
				} else {
					for (FloatMatrix input : streamerTrain) {
						TrainNetwork.train(network, input, streamerTrain.getOutput(input), LEARNING_RATE);
					}

				}
				
			}

			float totalScore = 0;
			float count = 0;
			for (FloatMatrix item : streamerTest) {
				count++;

				FloatMatrix actual = network.io(item);
				FloatMatrix expected = streamerTest.getOutput(item);
				float score = score(actual, expected);
				// System.out.println(count + "] Score: " + score + " > Actual:
				// " + network.io(item) + " > Expected: "
				// + streamerTest.getOutput(item));
				totalScore += score;
			}

			System.out.println("Done  " + totalScore * 100f / count);
		}
		
		dw.close();
	}

	private float score(FloatMatrix actual, FloatMatrix expected) {
		float score = 0f;
		if (actual.length == expected.length) {
			for (int i = 0; i < actual.length; i++) {
				if (Math.abs(actual.get(i, 0) - expected.get(i, 0)) <= 0.1) {
					score += 1;
				} else {
					score += 0;
				}
			}
			return score / actual.length;
		}

		return score;
	}

}
