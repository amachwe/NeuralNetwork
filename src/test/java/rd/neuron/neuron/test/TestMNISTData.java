package rd.neuron.neuron.test;

import java.io.IOException;

import org.canova.image.mnist.MnistLabelFile;
import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.ConsoleDataWriter;
import rd.data.DataStreamer;
import rd.data.MnistImageFile;
import rd.data.MnistToDataStreamer;
import rd.neuron.neuron.FullyRandomLayerBuilder;
import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.SimpleNetwork;
import rd.neuron.neuron.SimpleNetworkPerformanceEvaluator;

public class TestMNISTData {

	private final int EPOCHS = 1000000;
	private final boolean SGD = true;
	private final float LEARNING_RATE = 0.01f;

	@Test
	public void doMNISTTest() throws IOException {
		DataStreamer streamerTrain = MnistToDataStreamer.createStreamer("d:\\ml stats\\mnist\\t10k-images.idx3-ubyte",
				"d:\\ml stats\\mnist\\t10k-labels.idx1-ubyte");
		System.out.println("Training Streamer Ready!");

		DataStreamer streamerTest = MnistToDataStreamer.createStreamer("d:\\ml stats\\mnist\\train-images.idx3-ubyte",
				"d:\\ml stats\\mnist\\train-labels.idx1-ubyte");
		System.out.println("Testing Streamer Ready!");

		SimpleNetwork network = new SimpleNetwork(new FullyRandomLayerBuilder(0.5f, 1f), Function.LOGISTIC, 784, 100,
				10);

		SimpleNetworkPerformanceEvaluator snpm = new SimpleNetworkPerformanceEvaluator(new ConsoleDataWriter());
		// Back Prop
		for (int i = 0; i < EPOCHS; i++) {
			if ((i * 100f / EPOCHS) % 10 == 0) {
				System.out.println(i * 100f / EPOCHS);

			}
			if (!SGD) {
				for (FloatMatrix item : streamerTrain) {

					FloatMatrix outputLayerNewWts[] = network.trainOutputLayer(LEARNING_RATE,
							streamerTrain.getOutput(item), network.io(item));
					FloatMatrix hiddenLayerNewWts[] = network.trainHiddenLayer(0, LEARNING_RATE,
							streamerTrain.getOutput(item), network.io(item), item);

					network.setOutputWeights(outputLayerNewWts[0]);
					network.setOutputBias(outputLayerNewWts[1]);
					network.setWeights(0, hiddenLayerNewWts[0]);
					network.setBias(0, hiddenLayerNewWts[1]);
				}
			} else {
				FloatMatrix item = streamerTrain.getRandom();

				FloatMatrix outputLayerNewWts[] = network.trainOutputLayer(LEARNING_RATE, streamerTrain.getOutput(item),
						network.io(item));
				FloatMatrix hiddenLayerNewWts[] = network.trainHiddenLayer(0, LEARNING_RATE,
						streamerTrain.getOutput(item), network.io(item), item);

				network.setOutputWeights(outputLayerNewWts[0]);
				network.setOutputBias(outputLayerNewWts[1]);
				network.setWeights(0, hiddenLayerNewWts[0]);
				network.setBias(0, hiddenLayerNewWts[1]);

			}
			// snpm.evaluate(network, streamer);
		}

		float totalScore = 0;
		float count = 0;
		for (FloatMatrix item : streamerTest) {
			count++;
			FloatMatrix actual = network.io(item);
			FloatMatrix expected = streamerTest.getOutput(item);
			float score = score(actual, expected);
			System.out.println(count + "] Score: " + score + " > Actual: " + network.io(item) + "  > Expected: "
					+ streamerTest.getOutput(item));
			totalScore += score;
		}

		System.out.println(totalScore * 100f / count);
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
