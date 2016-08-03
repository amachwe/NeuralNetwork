package rd.data.test;

import static org.junit.Assert.*;

import org.junit.Test;

import rd.data.ConfusionMatrix;

public class TestConfusionMatrix {

	@Test
	public void doTest()
	{
		ConfusionMatrix cm = new ConfusionMatrix();
		cm.incTP();
		cm.incTP();
		cm.incTN();
		cm.incTN();
		cm.incFP();
		cm.incFP();
		cm.incFN();
		cm.incFN();
		cm.incFN();
		System.out.println(cm);
		assertTrue(cm.getAccuracy()==0.44444445f);
		assertTrue(cm.getPrecision()==0.4f);
		assertTrue(cm.getRecall()==0.5f);
		
	}
}
