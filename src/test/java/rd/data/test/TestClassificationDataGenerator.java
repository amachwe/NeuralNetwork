package rd.data.test;

import static org.junit.Assert.*;

import org.junit.Test;

import rd.data.ClassificationDataGenerator;
import rd.data.DataStreamer;
/**
 * Data Classification Test
 * @author azahar
 *
 */
public class TestClassificationDataGenerator {

	@Test
	public void doTest() {
		/**
		 * Spread for Clusters;  Number of Clusters: 2
		 */
		float[] spread = new float[] { 1f, 1.1f };
		/**
		 * Centre Points - input length = 3
		 */
		float[][] centres = new float[][] { { 2.1f, 2.4f, 3.4f }, { 0.2f, 0.1f, 0.3f } };
		int[] numberOfInstances = new int[] { 500, 1000 };
		DataStreamer ds = ClassificationDataGenerator.generate(spread, centres, numberOfInstances, false,
				ClassificationDataGenerator.Shape.Curve);
		ds.toCsvStream().forEach(item -> assertTrue(item!=null));
	}
}
