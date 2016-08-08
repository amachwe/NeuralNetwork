package rd.learning.statistics.test;

import java.io.IOException;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.DataStreamer;
import rd.data.MnistToDataStreamer;
import rd.learning.statistics.MulticlassLogReg;

public class TestMulticlassLogRegMNIST {
	/**
	 * MNIST: Files for Labels/Images - Training and Testing - change to run
	 * tests
	 */
	private static final String D_ML_STATS_MNIST_TRAIN_LABELS = "data\\train-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_TRAIN_IMAGES = "data\\train-images.idx3-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_LABELS = "data\\t10k-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_IMAGES = "data\\t10k-images.idx3-ubyte";

	private static final int EPOCHS = 100;

	/**
	 * MNIST Test
	 * 
	 * @throws IOException
	 */
	@Test
	public void doMNISTTest() throws IOException {
		DataStreamer streamerTrain = MnistToDataStreamer.createStreamer(D_ML_STATS_MNIST_T10K_IMAGES,
				D_ML_STATS_MNIST_T10K_LABELS);
		System.out.println("Training Streamer Ready!");

		DataStreamer streamerTest = MnistToDataStreamer.createStreamer(D_ML_STATS_MNIST_TRAIN_IMAGES,
				D_ML_STATS_MNIST_TRAIN_LABELS);
		System.out.println("Testing Streamer Ready!");
		MulticlassLogReg mnistLogReg = new MulticlassLogReg(784, 10);
		for (int i = 0; i < EPOCHS; i++) {
			mnistLogReg.train(buildBatch(streamerTrain, 10), 0.15f);

		}

		System.out.println(error(streamerTest, mnistLogReg));
		float totalScore = 0;
		float count = 0;
		for (FloatMatrix item : streamerTest) {
			count++;

			FloatMatrix actual = mnistLogReg.predict(item);
			FloatMatrix expected = streamerTest.getOutput(item);
			float score = score(actual, expected);
			totalScore += score;
		}
		float finalScore = totalScore * 100f / count;

		System.out.println("Done  " + finalScore);
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

	/**
	 * Build Batch from Streamer
	 * 
	 * @param ds
	 *            - original streamer
	 * @param miniBatch
	 *            - mini batch size
	 * @return Batch streamer
	 */
	private DataStreamer buildBatch(DataStreamer ds, int miniBatch) {
		DataStreamer batch = new DataStreamer(ds.getRandom().length, ds.getOutput(ds.getRandom()).length);
		for (int i = 0; i < miniBatch; i++) {
			FloatMatrix item = ds.getRandom();

			batch.add(item.toArray(), ds.getOutput(item).toArray());
		}
		return batch;
	}

	/**
	 * Calculate Prediction Error in model
	 * 
	 * @param ds
	 * @param model
	 * @return
	 */
	private float error(DataStreamer ds, MulticlassLogReg model) {
		float error = 0;
		float count = 0;
		for (FloatMatrix item : ds) {
			FloatMatrix output = ds.getOutput(item);
			FloatMatrix predict = model.predict(item);
			System.out.println(output + "    " + predict);
			error += Math.abs((predict.sub(output).sum() / (float) predict.length));
			count++;
		}

		return error / count;
	}
}
