package rd.neuron.neuron.test;

import static org.junit.Assert.assertTrue;

import java.io.IOException;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.DataStreamer;
import rd.data.DataWriter;
import rd.data.DeltaInspector;
import rd.data.MongoDeltaWriter;
import rd.data.NotMNISTToDataStreamer;
import rd.neuron.neuron.FullyRandomLayerBuilder;
import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.SimpleNetwork;
import rd.neuron.neuron.TrainNetwork;

/**
 * Not MNIST Training Artificial Neural Network: Input -> Single Hidden ->
 * Output
 * 
 * Get the dataset from here: http://yaroslavvb.blogspot.co.uk/2011/09/notmnist-dataset.html
 * @author azahar
 *
 */
public class TestNotMNISTData {

	// mini-batch size is 1

	//Sample rate to 0 to disable performance data sampling.
	private static final double SAMPLE_RATE = 0;
	private static final int OUTPUT_LAYER_SIZE = 26;
	private static final int HIDDEN_LAYER_SIZE = 300;
	private static final int INPUT_LAYER_SIZE = 784;
	/**
	 * Files for Labels/Images - Training and Testing - change to run tests
	 */
	private static final String D_ML_STATS_NOT_MNIST_DATASET_FULL = "d:\\ml stats\\notMnist\\notMnist_large";

	/**
	 * Model constants
	 */
	private final int EPOCHS = 100000;
	//How many times to repeat the train-evaluate cycle
	private final int REPEATS = 1;
	// Use Stochastic Descent or not
	private final boolean useSGD = true;
	private final float LEARNING_RATE = 0.06f;

	@Test
	public void doMNISTTest() throws IOException {
		DataStreamer fullSet = NotMNISTToDataStreamer.createStreamer(D_ML_STATS_NOT_MNIST_DATASET_FULL, 784,8);
		DataStreamer streamers[] = fullSet.split(0.4f);
		DataStreamer streamerTrain = streamers[0], streamerTest = streamers[1];

		System.out.println("Streamer Ready!");
		// Write Run Performance information to Mongo DB
		DataWriter dw = new MongoDeltaWriter("localhost", 27017, "neural", "sgd_backprop_notmnist");

		for (int round = 0; round < REPEATS; round++) {
			SimpleNetwork network = new SimpleNetwork(new FullyRandomLayerBuilder(0.5f, 1f), Function.LOGISTIC,
					INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
			network.activateDeepNetworkInspection(new DeltaInspector(SAMPLE_RATE, dw));
			System.out.println(round);
			// Back Prop
			for (int i = 0; i < EPOCHS; i++) {
				if ((i * 100f / EPOCHS) % 10 == 0) {
					System.out.println(i * 100f / EPOCHS);

				}
				if (useSGD) {
					FloatMatrix input = streamerTrain.getRandom();
					TrainNetwork.trainBackprop(network, input, streamerTrain.getOutput(input), LEARNING_RATE);

				} else {
					for (FloatMatrix input : streamerTrain) {
						TrainNetwork.trainBackprop(network, input, streamerTrain.getOutput(input), LEARNING_RATE);
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
				totalScore += score;
			}
			float finalScore = totalScore * 100f / count;
			assertTrue(finalScore > 0.89);
			System.out.println("Done  " + finalScore);
		}

		dw.close();
	}

	/**
	 * Score the network
	 * 
	 * @param actual
	 *            - actual output
	 * @param expected
	 *            - expected output
	 * @return
	 */
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
