package rd.neuron.neuron.test;

import static org.junit.Assert.*;

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
/**
 * MNIST Training Artificial Neural Network: Input -> Single Hidden -> Output
 * @author azahar
 *
 */
public class TestMNISTData {

	//mini-batch size is 1
	
	private static final double SAMPLE_RATE = 1e-6;
	private static final int OUTPUT_LAYER_SIZE = 10;
	private static final int HIDDEN_LAYER_SIZE = 15;
	private static final int INPUT_LAYER_SIZE = 784;
	/**
	 * Files for Labels/Images - Training and Testing - change to run tests
	 */
	private static final String D_ML_STATS_MNIST_TRAIN_LABELS = "d:\\ml stats\\mnist\\train-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_TRAIN_IMAGES = "d:\\ml stats\\mnist\\train-images.idx3-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_LABELS = "d:\\ml stats\\mnist\\t10k-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_IMAGES = "d:\\ml stats\\mnist\\t10k-images.idx3-ubyte";
	
	/**
	 * Model constants
	 */
	private final int EPOCHS = 1000000;
	private final int REPEATS = 1;
	//Use Stochastic Descent or not
	private final boolean useSGD = false;
	private final float LEARNING_RATE = 0.05f;

	@Test
	public void doMNISTTest() throws IOException {
		DataStreamer streamerTrain = MnistToDataStreamer.createStreamer(D_ML_STATS_MNIST_T10K_IMAGES,
				D_ML_STATS_MNIST_T10K_LABELS);
		System.out.println("Training Streamer Ready!");

		DataStreamer streamerTest = MnistToDataStreamer.createStreamer(D_ML_STATS_MNIST_TRAIN_IMAGES,
				D_ML_STATS_MNIST_TRAIN_LABELS);
		System.out.println("Testing Streamer Ready!");
		
		DataWriter dw = new MongoDeltaWriter("localhost",27017,"neural","sgd_backprop");
		

		for (int round = 0; round < REPEATS; round++) {
			SimpleNetwork network = new SimpleNetwork(new FullyRandomLayerBuilder(0.5f, 1f), Function.LOGISTIC, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE,
					OUTPUT_LAYER_SIZE);
			network.activateDeepNetworkInspection(new DeltaInspector(SAMPLE_RATE,dw));

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
				totalScore += score;
			}
			float finalScore = totalScore * 100f / count;
			assertTrue(finalScore>0.89);
			System.out.println("Done  " + finalScore);
		}
		
		dw.close();
	}

	/**
	 * Score the network 
	 * @param actual - actual output
	 * @param expected - expected output
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
