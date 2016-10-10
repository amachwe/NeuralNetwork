package rd.neuron.neuron.test;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.List;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.DataStreamer;
import rd.data.DataWriter;
import rd.data.MnistToDataStreamer;
import rd.data.MongoDeltaWriter;
import rd.neuron.neuron.Layer.Function;
import rd.neuron.neuron.LayerIf;
import rd.neuron.neuron.Propagate;
import rd.neuron.neuron.RecipeNetworkBuilder;
import rd.neuron.neuron.StochasticNetwork;

/**
 * MNIST Training Artificial Neural Network: Input -> Single Hidden -> Output
 * 
 * Get the dataset from here: http://yann.lecun.com/exdb/mnist/
 * 
 * @author azahar
 *
 */
public class TestRecipeMNISTData {

	// mini-batch size is 1


	/**
	 * Files for Labels/Images - Training and Testing - change to run tests
	 */
	private static final String D_ML_STATS_MNIST_TRAIN_LABELS = "data\\train-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_TRAIN_IMAGES = "data\\train-images.idx3-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_LABELS = "data\\t10k-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_IMAGES = "data\\t10k-images.idx3-ubyte";

	/**
	 * Model constants
	 */
	private final int EPOCHS = 10000;
	// How many times to repeat the train-evaluate cycle
	private final int REPEATS = 1;
	// Use Stochastic Descent or not
	private final boolean useSGD = true;


	@Test
	public void doMNISTTest() throws IOException {
		DataStreamer streamerTrain = MnistToDataStreamer.createStreamer(D_ML_STATS_MNIST_T10K_IMAGES,
				D_ML_STATS_MNIST_T10K_LABELS);
		System.out.println("Training Streamer Ready!");

		DataStreamer streamerTest = MnistToDataStreamer.createStreamer(D_ML_STATS_MNIST_TRAIN_IMAGES,
				D_ML_STATS_MNIST_TRAIN_LABELS);
		System.out.println("Testing Streamer Ready!");

		String recipe = "RANDOM 784 300\nRANDOM 300 300\nRANDOM 300 10";

		List<LayerIf> network = RecipeNetworkBuilder.build(recipe);
		StochasticNetwork nw = new StochasticNetwork(network, Function.LOGISTIC, Function.LOGISTIC);


		for (int round = 0; round < REPEATS; round++) {
			
			for(FloatMatrix in:streamerTrain)
			{
				//nw.preTrain(in);
			}

			// Back Prop
			for (int i = 0; i < EPOCHS; i++) {
				if ((i * 100f / EPOCHS) % 10 == 0) {
					System.out.println(i * 100f / EPOCHS);

				}
				if (useSGD) {
					FloatMatrix input = streamerTrain.getRandom();
					
					nw.fineTuneOutputLayer(streamerTrain.getOutput(input), Propagate.up(input, network), input);

				} else {
					for (FloatMatrix input : streamerTrain) {
						nw.fineTuneOutputLayer(streamerTrain.getOutput(input), Propagate.up(input, network), input);
					}

				}

			}

			float totalScore = 0;
			float count = 0;
			for (FloatMatrix item : streamerTest) {
				count++;

				FloatMatrix actual = Propagate.up(item,network);
				FloatMatrix expected = streamerTest.getOutput(item);
				float score = score(actual, expected);
				totalScore += score;
			}
			float finalScore = totalScore * 100f / count;
			assertTrue(finalScore > 0.89);
			System.out.println("Done  " + finalScore);
		}


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
