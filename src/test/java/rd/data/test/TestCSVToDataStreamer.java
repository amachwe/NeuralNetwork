package rd.data.test;

import static org.junit.Assert.assertTrue;

import java.io.IOException;

import org.jblas.FloatMatrix;
import org.junit.Test;

import rd.data.CSVToDataStreamer;
import rd.data.ClassHandler;
import rd.data.DataStreamer;

public class TestCSVToDataStreamer {

	@Test
	public void doTest() throws IOException {
		CSVToDataStreamer csv = new CSVToDataStreamer("data//iris.csv");

		DataStreamer ds = csv.getDataStreamer(4, 1,new ClassHandler[]{new ClassHandler()});
		for (FloatMatrix input : ds) {
			assertTrue(input != null);
			assertTrue(ds.getOutput(input) != null);
		}
	}
}
