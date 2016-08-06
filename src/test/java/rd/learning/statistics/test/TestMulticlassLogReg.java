package rd.learning.statistics.test;

import java.io.IOException;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.CSVToFlatClassDataStreamer;
import rd.data.ClassHandler;
import rd.data.DataStreamer;
import rd.learning.statistics.MulticlassLogReg;

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
				// System.out.println(prediction + " -- " +
				// iris.getOutput(input));
				if (prediction.get(0) - iris.getOutput(input).get(0) < 0.1) {
					score++;
				}
				count++;
			}

			System.out.println(i + "," + trainSize + "," + testSize + "," + (score * 100 / count));
		}

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
