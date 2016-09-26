package performance.tests;

import org.jblas.FloatMatrix;

//Test the performance of toString on FloatMatrix.
public class FloatMatrixToString {

	public static void main(String... args) {
		int repeats = 5;
		
		FloatMatrix toString = new FloatMatrix(1,repeats), toListToString = new FloatMatrix(1,repeats);
		//Run the tests
		for (int i = 0; i < repeats; i++) {
			
			toListToString.put(0,i,doToListToStringTest());
			toString.put(0,i,doToStringTest());
		}
		
		//Average values printed here.
		System.out.println("AVG: ToString (secs): "+toString.mean());
		System.out.println("AVG: ToList ToString (secs): "+toListToString.mean());
	}

	//Convert to List then call to String on the List
	private static float doToListToStringTest() {
		FloatMatrix temp = FloatMatrix.rand(100, 100);
		int runs = 1000;

		long start = System.currentTimeMillis();
		for (int i = 0; i < runs; i++) {
			temp.elementsAsList().toString();
		}
		float timeTaken = 0;
		System.out.println("toList, toString - Time staken (secs): "
				+ (timeTaken = ((System.currentTimeMillis() - start) / 1000)));
		return timeTaken;
	}

	//Call to String directly on Float Matrix
	private static float doToStringTest() {
		FloatMatrix temp = FloatMatrix.rand(100, 100);
		int runs = 1000;
		long start = System.currentTimeMillis();

		for (int i = 0; i < runs; i++) {
			temp.toString();
		}
		float timeTaken = 0;
		System.out.println(
				"toString - Time staken (secs): " + (timeTaken = ((System.currentTimeMillis() - start) / 1000)));
		return timeTaken;
	}
}
