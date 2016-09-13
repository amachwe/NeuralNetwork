package rd.data.test;

import static org.junit.Assert.*;

import java.util.Arrays;

import org.junit.Test;

import rd.data.NotMNISTToDataStreamer;

public class TestNotMnistLabel {
@Test
public void doLabelTest()
{
	
	for (char a = 'a'; a <= 'z'; a++) {
		float[] lc = NotMNISTToDataStreamer.getOneHotEncoding(a);
		float[] uc = NotMNISTToDataStreamer.getOneHotEncoding(Character.toUpperCase(a));
		System.out.println("\n\n" + a + ": " + Arrays.toString(lc) + "\n" + Character.toUpperCase(a) + ": "
				+ Arrays.toString(uc));
		assertTrue(Arrays.equals(lc, uc));

	}
}
}
