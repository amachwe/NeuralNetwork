package rd.learning.statistics.test;

import java.io.IOException;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.CSVToFlatClassDataStreamer;
import rd.data.ClassHandler;
import rd.data.DataStreamer;
import rd.data.MnistToDataStreamer;
import rd.learning.statistics.MulticlassLogReg;

/**
 * Test Multi Class Logistic Regression
 * 
 * @author azahar
 *
 */
public class TestMulticlassLogReg {

	/**
	 * MNIST: Files for Labels/Images - Training and Testing - change to run
	 * tests
	 */
	private static final String D_ML_STATS_MNIST_TRAIN_LABELS = "data\\train-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_TRAIN_IMAGES = "data\\train-images.idx3-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_LABELS = "data\\t10k-labels.idx1-ubyte";
	private static final String D_ML_STATS_MNIST_T10K_IMAGES = "data\\t10k-images.idx3-ubyte";

	private static final int EPOCHS = 10000;

	/**
	 * Stand alone test with sample data
	 */
	@Test
	public void doTest() {
		MulticlassLogReg reg = new MulticlassLogReg(2, 2);

		DataStreamer ds = new DataStreamer(2, 2);
		ds.add(new float[] { 0.1f, 0.2f }, 1, 0);
		ds.add(new float[] { 1f, 2f }, 0, 1);
		ds.add(new float[] { 0.4f, 0.3f }, 1, 0);
		ds.add(new float[] { 3f, 2f }, 0, 1);
		ds.add(new float[] { 0.3f, 0.2f }, 1, 0);
		ds.add(new float[] { 3f, 3f }, 0, 1);

		for (int i = 0; i < EPOCHS; i++) {

			reg.train(ds, 0.02f);

		}

		System.out.println(error(ds, reg));

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

	/**
	 * Iris Data Set Test
	 */
	@Test
	public void doIrisTest() throws IOException {
		DataStreamer iris = (new CSVToFlatClassDataStreamer("data//iris.csv")).getFlatClassDataStreamer(4, 2,
				new ClassHandler());
		DataStreamer[] split = iris.split(0.6f);
		System.out.println("Total: "+iris.getNumberOfUniqueInputs()+"     Split > Train: "+split[0].getNumberOfUniqueInputs()+"  Test: "+split[1].getNumberOfUniqueInputs());
		MulticlassLogReg reg = new MulticlassLogReg(4, 2);
		for (int i = 0; i < 2000; i++) {

			reg.train(buildBatch(split[0], 10), 0.02f);

		}
		float count = 0, score = 0;
		for (FloatMatrix input : split[1]) {
			FloatMatrix prediction = reg.predict(input);
			System.out.println(prediction + " -- " + iris.getOutput(input));
			if (prediction.get(0) - iris.getOutput(input).get(0) < 0.1) {
				score++;
			}
			count++;
		}

		System.out.println(score * 100 / count);

	}

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
