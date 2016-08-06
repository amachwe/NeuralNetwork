package rd.learning.statistics.test;

import java.io.IOException;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.CSVToFlatClassDataStreamer;
import rd.data.ClassHandler;
import rd.data.ClassificationDataGenerator;
import rd.data.ConfusionMatrix;
import rd.data.DataStreamer;
import rd.learning.statistics.MulticlassLogReg;
import rd.neuron.neuron.perceptron.Perceptron;

/**
 * Test Multi Class Logistic Regression
 * 
 * @author azahar
 *
 */
public class TestMulticlassLogReg {

	private static final int TRIES = 1000;
	private static final int EPOCHS = 10000;

	/**
	 * OR Test which should work very well as OR function data is linearly separable
	 */
	@Test
	public void doORTest() {
		ConfusionMatrix cm = new ConfusionMatrix();
		DataStreamer input = new DataStreamer(2, 1);
		// OR data set

		input.add(new float[] { 0f, 0.0f }, 0f);
		input.add(new float[] { 1f, 0.0f }, 1f);
		input.add(new float[] { 0f, 1.0f }, 1f);
		input.add(new float[] { 1f, 1.0f }, 1f);

		float[] LEARNING_RATE = new float[] { 0.5f, 0.2f, 0.09f, 0.08f, 0.07f, 0.06f, 0.05f, 0.02f, 0.002f };
		for (int j = 0; j < LEARNING_RATE.length; j++) {
			MulticlassLogReg reg = new MulticlassLogReg(2, 1);
			for (int i = 0; i < EPOCHS; i++) {

				reg.train(input, LEARNING_RATE[j]);

			}
			for (FloatMatrix item : input) {

				int actual = (int) input.getOutput(item).get(0);
				float calculated = reg.predict(item).get(0);
				System.out.println(item + "  " + reg.predict(item));
				if (actual == calculated ) {
					cm.incTP();
			
				} else if (actual < calculated ) {
					cm.incFP();
				} else if (actual > calculated ) {
					cm.incFN();
				}
			}
			System.out.println("OR: " + LEARNING_RATE[j] + "," + cm);

		}
	}
	/**
	 * XOR Test - should fail badly as XOR data is not linearly separable.
	 */
	@Test
	public void doXORTest() {
		ConfusionMatrix cm = new ConfusionMatrix();
		DataStreamer input = new DataStreamer(2, 1);
		// XOR data set

		input.add(new float[] { 0f, 0.0f }, 1f);
		input.add(new float[] { 1f, 0.0f }, 0f);
		input.add(new float[] { 0f, 1.0f }, 0f);
		input.add(new float[] { 1f, 1.0f }, 1f);

		float[] LEARNING_RATE = new float[] { 0.5f, 0.2f, 0.09f, 0.08f, 0.07f, 0.06f, 0.05f, 0.02f, 0.002f };
		for (int j = 0; j < LEARNING_RATE.length; j++) {
			MulticlassLogReg reg = new MulticlassLogReg(2, 1);
			for (int i = 0; i < EPOCHS; i++) {

				reg.train(input, LEARNING_RATE[j]);

			}
			for (FloatMatrix item : input) {

				int actual = (int) input.getOutput(item).get(0);
				float calculated = reg.predict(item).get(0);
				System.out.println(item + "  " + reg.predict(item));
				if (actual == calculated ) {
					cm.incTP();
			
				} else if (actual < calculated ) {
					cm.incFP();
				} else if (actual > calculated ) {
					cm.incFN();
				}
			}
			System.out.println("XOR: " + LEARNING_RATE[j] + "," + cm);

		}
	}

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
	 * Do Generated Data Set Test 2000 instances
	 */
	@Test
	public void doGeneratedDataSet() {
		/**
		 * Spread for Clusters; Number of Clusters: 2
		 */
		float[] spread = new float[] { 1f, 1.1f };
		/**
		 * Centre Points - input length = 3
		 */
		float[][] centres = new float[][] { { 3.1f, 3.4f, 3.4f }, { 0.2f, 0.1f, 0.3f } };
		int[] numberOfInstances = new int[] { 500, 1000 };
		DataStreamer ds = ClassificationDataGenerator.generate(spread, centres, numberOfInstances, true,
				ClassificationDataGenerator.Shape.Square);
		for (int i = 0; i < TRIES; i++) {

			DataStreamer[] split = ds.split(0.3f + (float) (0.3 * Math.random()));

			int trainSize = split[0].getNumberOfUniqueInputs();
			int testSize = split[1].getNumberOfUniqueInputs();

			MulticlassLogReg reg = new MulticlassLogReg(3, 2);
			for (int iter = 0; iter < 2000; iter++) {

				reg.train(buildBatch(split[0], 10), 0.02f);

			}
			float count = 0, score = 0;
			for (FloatMatrix input : split[1]) {
				FloatMatrix prediction = reg.predict(input);

				if (prediction.get(0) - ds.getOutput(input).get(0) < 0.1) {
					score++;
				}
				count++;
			}

			System.out.println(i + "," + trainSize + "," + testSize + "," + (score * 100 / count));
		}

		System.out.println("Data Generated Done");

	}

	/**
	 * Iris Data Set Test
	 */
	@Test
	public void doIrisTest() throws IOException {
		for (int i = 0; i < TRIES; i++) {
			DataStreamer iris = (new CSVToFlatClassDataStreamer("data//iris.csv")).getFlatClassDataStreamer(4, 2,
					new ClassHandler());
			DataStreamer[] split = iris.split(0.3f + (float) (0.3 * Math.random()));

			int trainSize = split[0].getNumberOfUniqueInputs();
			int testSize = split[1].getNumberOfUniqueInputs();

			MulticlassLogReg reg = new MulticlassLogReg(4, 2);
			for (int iter = 0; iter < 2000; iter++) {

				reg.train(buildBatch(split[0], 10), 0.02f);

			}
			float count = 0, score = 0;
			for (FloatMatrix input : split[1]) {
				FloatMatrix prediction = reg.predict(input);

				if (prediction.get(0) - iris.getOutput(input).get(0) < 0.1) {
					score++;
				}
				count++;
			}

			System.out.println(i + "," + trainSize + "," + testSize + "," + (score * 100 / count));
		}
		System.out.println("Data Iris Done");
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

}
