package rd.data.test;

import static org.junit.Assert.*;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.junit.Test;

import rd.data.DataStreamer;
import rd.data.MnistToDataStreamer;

/**
 * Test Mnist Data Loader
 * 
 * @author azahar
 *
 */
public class TestMnistDataLoader {

	@Test
	public void doCountTest() throws FileNotFoundException, IOException {
		DataStreamer streamerTest = MnistToDataStreamer.createStreamer("data\\train-images.idx3-ubyte",
				"data\\train-labels.idx1-ubyte");
		streamerTest.forEach(matrix -> assertNotNull(matrix));
	}
}
