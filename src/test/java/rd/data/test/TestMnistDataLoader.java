package rd.data.test;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.junit.Test;

import rd.data.DataStreamer;
import rd.data.MnistToDataStreamer;

public class TestMnistDataLoader {

	@Test
	public void doCountTest() throws FileNotFoundException, IOException
	{
		DataStreamer streamerTest = MnistToDataStreamer.createStreamer("d:\\ml stats\\mnist\\train-images.idx3-ubyte",
				"d:\\ml stats\\mnist\\train-labels.idx1-ubyte");
	}
}
